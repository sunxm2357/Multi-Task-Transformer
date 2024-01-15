# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)


import os, json, imageio
import numpy as np
from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import to_cuda, get_output, mkdir_if_missing
import torch
from tqdm import tqdm
from utils.test_utils import test_phase
import math
import torch.distributed as dist
from copy import deepcopy
import time

repr_metric_keys = {'semseg': 'mIoU', 'depth': 'rmse', 'normals': 'mean', 'edge': 'loss'}
large_better = {'semseg': 1, 'depth': -1 , 'normals': -1, 'edge': -1}
stl_performance = {'semseg': 43.63, 'depth': 0.6145159115 , 'normals': 21.15938833, 'edge': 0.04755854912}

def get_task_repr_metric(metrics, task):
  if task in repr_metric_keys:
    return metrics[task][repr_metric_keys[task]] * large_better[task]
  else:
    raise ValueError('Does not find representative metrics')


def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)


def train_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count):
    """ Vanilla training with fixed loss weights """
    model.train() 

    # For visualization of 3ddet in each epoch
    if '3ddet' in p.TASKS.NAMES:
        train_save_dirs = {task: os.path.join(p['save_dir'], 'train', task) for task in ['3ddet']}
        for save_dir in train_save_dirs.values():
            mkdir_if_missing(save_dir)

    for i, cpu_batch in enumerate(tqdm(train_loader)):
        # Forward pass
        batch = to_cuda(cpu_batch)
        images = batch['image'] 
        output = model(images)
        iter_count += 1
        
        # Measure loss
        loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES)
        # get learning rate
        lr = scheduler.get_lr()
        loss_dict['lr'] = torch.tensor(lr[0])

        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), **p.grad_clip_param)
        optimizer.step()
        scheduler.step()

        # vis training sample 3ddet
        if '3ddet' in p.TASKS.NAMES and i==0:
            from detection_toolbox.det_tools import bbox2json, bbox2fig
            task = '3ddet'
            sample = cpu_batch
            inputs, meta = batch['image'], sample['meta']
            det_res_list = get_output(output['3ddet'], '3ddet', p=p, label=sample)
            bs = int(inputs.size()[0])
            K_matrixes = sample['meta']['K_matrix']
            cam_params = [{k: v[sa] for k, v in sample['bbox_camera_params'].items()} for sa in range(bs)]
            # get gt labels 
            gt_center_I = []
            gt_center_S = []
            gt_size_S = []
            gt_rotation_S = []
            gt_class = []
            for _i in range(bs):
                if type(sample['det_labels'][_i]) == dict:
                    gt_center_I.append(sample['det_labels'][_i]['center_I'].cpu().numpy())
                    gt_center_S.append(sample['det_labels'][_i]['center_S'].cpu().numpy())
                    gt_size_S.append(sample['det_labels'][_i]['size_S'].cpu().numpy())
                    gt_rotation_S.append(sample['det_labels'][_i]['rotation_S'].cpu().numpy())
                    gt_class.append(sample['det_labels'][_i]['label'])
                else:
                    gt_center_I.append(None)
                    gt_center_S.append(None)
                    gt_size_S.append(None)
                    gt_rotation_S.append(None)
                    gt_class.append(None)
            # save bbox predictions in cityscapes evaluation format
            for jj in range(bs):
                fname = 'b' + str(epoch) + '_' + meta['img_name'][jj]
                json_dict = bbox2json(det_res_list[jj], K_matrixes[jj], cam_params[jj])
                out_path = os.path.join(train_save_dirs['3ddet'], fname + '.json')
                with open(out_path, 'w') as outfile:
                    json.dump(json_dict, outfile)
                if True:
                    # visualization, but it takes time so we manually use it
                    box_no = len(det_res_list[jj]['img_bbox']['scores_3d'])
                    if box_no > 0:
                        gt_labels = [gt_class[jj], gt_center_I[jj], gt_center_S[jj], gt_size_S[jj], gt_rotation_S[jj]]
                        vis_fig = bbox2fig(p, inputs[jj].cpu(), det_res_list[jj], K_matrixes[jj], cam_params[jj], gt_labels)
                        imageio.imwrite(os.path.join(train_save_dirs[task], fname + '_' + str(box_no) + '.png'), vis_fig.astype(np.uint8))

        
        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            # return True, iter_count
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True 
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            if '3ddet' in curr_result.keys():
                curr_result.pop('3ddet')
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            # Checkpoint after evaluation
            print('Checkpoint starts at iter {}....'.format(iter_count))
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                        'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint'])
            print('Checkpoint finishs.')
            model.train() # set model back to train status

        if end_signal:
            return True, iter_count

    return False, iter_count


def train_phase_no_overlap_data(p, args, train_loaders, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer,
                tb_writer_test, iter_count):
    """ Vanilla training with fixed loss weights """
    model.train()

    # For visualization of 3ddet in each epoch
    if '3ddet' in p.TASKS.NAMES:
        train_save_dirs = {task: os.path.join(p['save_dir'], 'train', task) for task in ['3ddet']}
        for save_dir in train_save_dirs.values():
            mkdir_if_missing(save_dir)

    dataloader_len = [len(dataloader) for dataloader in train_loaders]
    min_epoch_len = min(dataloader_len)
    num_dataloaders = len(train_loaders)
    iter_dataloaders = [iter(dataloader)  for dataloader in train_loaders ]

    for i in tqdm(range(min_epoch_len)):
        optimizer.zero_grad()
        for t_id, iter_dataloader in enumerate(iter_dataloaders):
            # Forward pass
            cpu_batch = next(iter_dataloader)
            batch = to_cuda(cpu_batch)
            images = batch['image']
            output = model(images)

            # Measure loss
            loss_dict = criterion(output, batch, tasks=[p.TASKS.NAMES[t_id]])
            # get learning rate
            lr = scheduler.get_lr()
            loss_dict['lr'] = torch.tensor(lr[0])

            if tb_writer is not None:
                update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)

            loss_dict['total'].backward()
        iter_count += 1

        # Backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), **p.grad_clip_param)
        optimizer.step()
        scheduler.step()

        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            # return True, iter_count
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            if '3ddet' in curr_result.keys():
                curr_result.pop('3ddet')
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            # Checkpoint after evaluation
            print('Checkpoint starts at iter {}....'.format(iter_count))
            torch.save(
                {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(),
                 'epoch': epoch, 'iter_count': iter_count - 1}, p['checkpoint'])
            print('Checkpoint finishs.')
            model.train()  # set model back to train status

        if end_signal:
            return True, iter_count

    return False, iter_count


def our_affinity(model, optimizer, scheduler, criterion, affinity_data_loaders, test_dataloader,
                 args, p, iter_count):
  start_time0 = time.time()
  rank = dist.get_rank() if dist.is_initialized() else 0
  affinity_enumerators = []
  for data_loader in affinity_data_loaders:
    if dist.is_initialized():
      data_loader.sampler.set_epoch(iter_count)
    enumerator = enumerate(data_loader)
    affinity_enumerators.append(enumerator)
  # save the current state dict for model and optimizer
  save_model_state_dict = deepcopy(model.state_dict())
  save_optimizer_state_dict = deepcopy(optimizer.state_dict())
  save_scheduler_state_dict = deepcopy(scheduler.state_dict())
  num_task = len(affinity_enumerators)
  model.eval()

  aff_mat = to_cuda( torch.zeros(num_task, num_task))

  curr_result_before = test_phase(p, test_dataloader, model, criterion, None, num_batch=args.affinity_batches)
  start_time = time.time()
  print('evaluation time = ', time.time() - start_time)

  dist.barrier()
  for idx in range(num_task):
    metric_before = get_task_repr_metric(curr_result_before, p.TASKS.NAMES[idx])

    start_time = time.time()
    # look ahead for several steps with its own loss
    model.train()
    for _ in range(args.look_ahead_steps):
      optimizer.zero_grad()
      a_id, cpu_batch = next(affinity_enumerators[idx])
      if a_id == len(affinity_data_loaders[idx]) - 1:
        affinity_enumerators[idx] = enumerate(
            affinity_data_loaders[idx])

      batch = to_cuda(cpu_batch)
      images = batch['image']
      output = model(images)

      # Measure loss
      loss_dict = criterion(output, batch, tasks=[p.TASKS.NAMES[idx]])

      loss_dict['total'].backward()
      optimizer.step()
      scheduler.step()
    print('STL training time for affinity_steps = ', time.time()-start_time)
    start_time = time.time()
    dist.barrier()
    # self evaluation
    model.eval()
    curr_result_after_self = test_phase(p, test_dataloader, model, criterion, None, num_batch=args.affinity_batches)
    dist.barrier()
    if rank == 0:
        metric_after_self = get_task_repr_metric(curr_result_after_self, p.TASKS.NAMES[idx])
        print(
            '%d-%d: metric before=%f, metric after=%f, metric drop =%f' %
            (idx, idx, metric_before, metric_after_self,
             metric_after_self - metric_before))
    print('Evaluation time after STL for affinity_steps = ', time.time() - start_time)
    start_time = time.time()
    model.load_state_dict(save_model_state_dict)
    optimizer.load_state_dict(save_optimizer_state_dict)
    scheduler.load_state_dict(save_scheduler_state_dict)
    print('Evaluation time for reloading the model = ', time.time() - start_time)
    for idx2 in range(num_task):
      start_time = time.time()
      model.train()
      for _ in range(args.look_ahead_steps):
          optimizer.zero_grad()

          a_id, cpu_batch = next(affinity_enumerators[idx])
          if a_id == len(affinity_data_loaders[idx]) - 1:
              affinity_enumerators[idx] = enumerate(
                  affinity_data_loaders[idx])

          batch = to_cuda(cpu_batch)
          images = batch['image']
          output = model(images)

          # Measure loss
          loss_dict = criterion(output, batch, tasks=[p.TASKS.NAMES[idx]])

          loss_dict['total'].backward()

          a_id, cpu_batch = next(affinity_enumerators[idx])
          if a_id == len(affinity_data_loaders[idx]) - 1:
              affinity_enumerators[idx] = enumerate(
                  affinity_data_loaders[idx])

          batch = to_cuda(cpu_batch)
          images = batch['image']
          output = model(images)

          # Measure loss
          loss_dict = criterion(output, batch, tasks=[p.TASKS.NAMES[idx]])

          loss_dict['total'].backward()

          a_id, cpu_batch = next(affinity_enumerators[idx2])
          if a_id == len(affinity_data_loaders[idx2]) - 1:
            affinity_enumerators[idx2] = enumerate(
                affinity_data_loaders[idx2])

          batch = to_cuda(cpu_batch)
          images = batch['image']
          output = model(images)

          # Measure loss
          loss_dict = criterion(output, batch, tasks=[p.TASKS.NAMES[idx2]])

          loss_dict['total'].backward()

          optimizer.step()
          scheduler.step()
          dist.barrier()

      print('MTL training time for affinity steps  = ', time.time() - start_time)
      # evaluate the loss after looking ahead
      model.eval()
      curr_result_after_joint = test_phase(p, test_dataloader, model, criterion, None,
                                          num_batch=args.affinity_batches)


      if rank == 0:
          metric_after_joint = get_task_repr_metric(curr_result_after_joint, p.TASKS.NAMES[idx])
          aff_mat[idx2, idx] = metric_after_joint - metric_after_self

          if args.affinity_normalized_by_lr:
            aff_mat[idx2, idx] = aff_mat[
                idx2, idx] / optimizer.param_groups[0]['lr']

          if args.affinity_normalized_by_stl:
            aff_mat[idx2, idx] = aff_mat[
                idx2, idx] / stl_performance[p.TASKS.NAMES[idx]]

          print(
              '%d-%d: metric before=%f, metric after=%f, metric drop =%f' %
              (idx2, idx, metric_before, metric_after_joint,
               metric_after_joint - metric_before))

          print('%d-%d: additional metric drop =%f' %
                       (idx2, idx,
                        metric_after_joint - metric_after_self))

      # restore the model and optimizer
      model.load_state_dict(save_model_state_dict)
      optimizer.load_state_dict(save_optimizer_state_dict)
      scheduler.load_state_dict(save_scheduler_state_dict)

  model.load_state_dict(save_model_state_dict)
  optimizer.load_state_dict(save_optimizer_state_dict)
  scheduler.load_state_dict(save_scheduler_state_dict)
  optimizer.zero_grad()
  model.train()
  print('Total time for affinity steps = ', time.time() - start_time0)
  return model, optimizer, scheduler, aff_mat


def train_phase_no_overlap_data_affinity(p, args, train_loaders, affinity_loaders, test_dataloader, model, criterion,
                                         optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count, aff_mat):
    """ Vanilla training with fixed loss weights """
    if aff_mat is not None:
        aff_mat = to_cuda(aff_mat)

    model.train()

    # For visualization of 3ddet in each epoch
    if '3ddet' in p.TASKS.NAMES:
        train_save_dirs = {task: os.path.join(p['save_dir'], 'train', task) for task in ['3ddet']}
        for save_dir in train_save_dirs.values():
            mkdir_if_missing(save_dir)

    dataloader_len = [len(dataloader) for dataloader in train_loaders]
    min_epoch_len = min(dataloader_len)
    iter_dataloaders = [iter(dataloader)  for dataloader in train_loaders ]

    num_task = len(train_loaders)
    # aff_mat_epoch = torch.zeros(
    #     (num_task, num_task, int(math.ceil(min_epoch_len / args.affinity_freq))))
    # aff_mat_epoch = to_cuda(aff_mat_epoch)

    for _ in tqdm(range(min_epoch_len)):
        if iter_count % args.affinity_freq == 0:
            model, optimizer, scheduler, aff_mat_tmp = our_affinity(model, optimizer, scheduler, criterion,
                                                                affinity_loaders, test_dataloader, args, p, iter_count)

            aff_mat_tmp = aff_mat_tmp.unsqueeze(-1)
            if aff_mat is None:
                aff_mat = aff_mat_tmp
            else:
                aff_mat = torch.cat([aff_mat, aff_mat_tmp], dim=-1)
        if iter_count % p.val_interval == 0 and args.local_rank == 0:
            mean_aff_mat = aff_mat.mean(dim=-1)
            for a_id in range(num_task):
                for b_id in range(num_task):
                    print('%d-%d: %f' % (a_id, b_id, mean_aff_mat[a_id, b_id]))

        print('start of opt')
        optimizer.zero_grad()
        for t_id, iter_dataloader in enumerate(iter_dataloaders):
            # Forward pass
            cpu_batch = next(iter_dataloader)
            batch = to_cuda(cpu_batch)
            images = batch['image']
            output = model(images)

            # Measure loss
            loss_dict = criterion(output, batch, tasks=[p.TASKS.NAMES[t_id]])
            print('loss computed ...')
            # get learning rate
            lr = scheduler.get_lr()
            loss_dict['lr'] = torch.tensor(lr[0])

            if tb_writer is not None:
                update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)

            loss_dict['total'].backward()
        iter_count += 1

        # Backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), **p.grad_clip_param)
        optimizer.step()
        scheduler.step()

        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            # return True, iter_count
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            if '3ddet' in curr_result.keys():
                curr_result.pop('3ddet')
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            # Checkpoint after evaluation
            print('Checkpoint starts at iter {}....'.format(iter_count))
            torch.save(
                {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(),
                 'epoch': epoch, 'iter_count': iter_count - 1, 'aff_mat': aff_mat}, p['checkpoint'])
            print('Checkpoint finishs.')
            model.train()  # set model back to train status

        if end_signal:
            return True, iter_count, aff_mat

    return False, iter_count, aff_mat


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]

def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    if 'semseg' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/semseg_miou', curr_result['semseg']['mIoU'], cur_iter)
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)
    if 'depth' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/depth_rmse', curr_result['depth']['rmse'], cur_iter)
    if '3ddet' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/3ddet_mDetectionScore', curr_result['3ddet']['mDetection_Score'], cur_iter)