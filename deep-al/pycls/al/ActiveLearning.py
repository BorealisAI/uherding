# Code is originally from Prateek Munjal et al. (https://arxiv.org/abs/2002.09564)
# from https://github.com/PrateekMunjal/TorchAL by Prateek Munjal which is licensed under MIT license
# You may obtain a copy of the License at
#
# https://github.com/PrateekMunjal/TorchAL/blob/master/LICENSE
#
####################################################################################

import os
import sys
sys.path.append(os.getcwd())

from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg, budget):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
        self.budget = budget

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                         val_dataloader=None, train_dataloader=None):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.budget > 0, "Expected a positive budgetSize"
        assert self.budget < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.budget)

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.budget)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "conf":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.confidence(budgetSize=self.budget, lSet=lSet,uSet=uSet \
                ,model=clf_model, dataset=trainDataset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(budgetSize=self.budget,lSet=lSet,uSet=uSet \
                ,model=clf_model, dataset=trainDataset, val_dataloader=val_dataloader)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(budgetSize=self.budget, lSet=lSet, uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower().startswith("badge"):
            al = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
            prime = "prime" in al
            herding = "herding" in al
            w_lset = "lset" in al
            simclr = "simclr" in al

            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.badge(
                budgetSize=self.budget, lSet=lSet, simclr=simclr,
                uSet=uSet, model=clf_model, dataset=trainDataset,
                prime=prime, herding=herding, w_lset=w_lset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower().startswith("wkmeans"):
            al = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
            herding = "herding" in al
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.weighted_kmeans(
                budgetSize=self.budget, lSet=lSet,
                total_uSet=uSet, model=clf_model, dataset=trainDataset, herding=herding)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["activeft"]:
            from .active_ft import ActiveFT

            oldmode = clf_model.training
            clf_model.eval()
            activeft = ActiveFT(self.cfg, lSet, uSet, budgetSize=self.budget,
                              clf_model=clf_model, dataset=trainDataset, dataObj=self.dataObj)
            activeSet, uSet = activeft.select_samples()
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj, lSet=lSet,
                                                uSet=uSet, clf_model=clf_model, dataset=trainDataset,
                                                budgetSize=self.budget)
            activeSet, uSet = coreSetSampler.query()
            clf_model.train()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("typiclust"):
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=self.budget,
                            is_scan=is_scan, dataset=trainDataset)
            activeSet, uSet = tpc.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower().startswith("uherding"):
            from .uherding import UHerding
            herding = UHerding(self.cfg, lSet, uSet, budgetSize=self.budget,
                               delta=self.cfg.ACTIVE_LEARNING.DELTA,
                               kernel=self.cfg.ACTIVE_LEARNING.KERNEL,
                               weighted=False,
                               clf_model=clf_model, dataset=trainDataset, dataObj=self.dataObj,
                               val_dataloader=val_dataloader, train_dataloader=train_dataloader)
            self.delta = herding.delta
            activeSet, uSet = herding.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["herding"]:
            from .herding import Herding
            herding = Herding(self.cfg, lSet, uSet, budgetSize=self.budget,
                              delta=self.cfg.ACTIVE_LEARNING.DELTA,
                              kernel=self.cfg.ACTIVE_LEARNING.KERNEL,
                              clf_model=clf_model, dataset=trainDataset, dataObj=self.dataObj)
            activeSet, uSet = herding.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=self.budget,
                            delta=self.cfg.ACTIVE_LEARNING.DELTA, dataset=trainDataset)
            activeSet, uSet = probcov.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=self.budget, uSet=uSet, clf_model=clf_model, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=self.budget, uSet=uSet, clf_models=supportingModels, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
            adv_sampler = AdversarySampler(
                cfg=self.cfg, dataObj=self.dataObj, budgetSize=self.budget)

            # Train VAE and discriminator first
            vae, disc, uSet_loader = adv_sampler.vaal_perform_training(lSet=lSet, uSet=uSet, dataset=trainDataset)

            # Do active sampling
            activeSet, uSet = adv_sampler.sample_for_labeling(vae=vae, discriminator=disc, \
                                unlabeled_dataloader=uSet_loader, uSet=uSet)
        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
