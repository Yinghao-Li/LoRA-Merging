import os.path as osp
import torch
import glob
from natsort import natsorted


def load_gradient_embeddings(grads_dir, checkpoint_ids, d_emb=8192):
    """
    Returns
    -------
    instance_ids : list
        List of instance ids.
    grad_embs : torch.Tensor
        The gradient embeddings, (n_instances, n_checkpoints, n_dims).
    """
    instance_ids = list()
    grad_embs_list = list()

    for checkpoint_idx in checkpoint_ids:

        checkpoint_instance_ids = list()
        grad_embs = None

        emb_paths = natsorted(glob.glob(osp.join(grads_dir, "split-*")))
        for emb_path in emb_paths:
            embs = torch.load(
                osp.join(emb_path, f"grads-dim.{d_emb}[{checkpoint_idx}]", f"all_unormed.pt"), weights_only=True
            )

            checkpoint_instance_ids += embs["instance_ids"]
            if grad_embs is None:
                grad_embs = embs["grads"]
            else:
                grad_embs = torch.concat([grad_embs, embs["grads"]], dim=0)

        if instance_ids:
            assert instance_ids == checkpoint_instance_ids, ValueError(
                "Instance ids must be the same across checkpoints."
            )
        instance_ids = checkpoint_instance_ids
        checkpoint_instance_ids = list()
        grad_embs_list.append(grad_embs)

    grad_embs = torch.stack(grad_embs_list, dim=1)

    return instance_ids, grad_embs
