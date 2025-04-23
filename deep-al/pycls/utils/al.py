# Code is originally from the Typiclust (https://arxiv.org/abs/2202.02794) and ProbCover (https://arxiv.org/abs/2205.11320) implementation
# from https://github.com/avihu111/TypiClust by Avihu Dekel and Guy Hacohen which is licensed under MIT license
# You may obtain a copy of the License at
#
# https://github.com/avihu111/TypiClust/blob/main/LICENSE
#
####################################################################################

class BudgetIterator:
    def __init__(self, cfg, num_train, init_num):
        self.cfg = cfg
        self.num_train = num_train
        self.init_num = init_num
        self.reset(cfg, num_train)

    def reset(self, cfg, num_train):
        if '_' in cfg.ACTIVE_LEARNING.BUDGET_SIZE:
            budget = cfg.ACTIVE_LEARNING.BUDGET_SIZE
        else:
            budget = int(cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        if isinstance(budget, str): # e.g. 10_10_100_9_1000_9
            splits = budget.split('_')
            assert len(splits) % 2 == 0

            tmp_max_iter = 0
            budget_list = []
            for i in range(0, len(splits), 2):
                budget_val = int(splits[i])
                num_iter = int(splits[i+1])

                budget_list += [budget_val] * num_iter
                tmp_max_iter += num_iter

            assert len(budget_list) == tmp_max_iter

        else:
            tmp_max_iter = cfg.ACTIVE_LEARNING.MAX_ITER
            assert tmp_max_iter is not None
            budget_list = [budget] * tmp_max_iter

        self.budget_list = budget_list
        self.index = 0

        # compute max_iter
        budget_count = 0
        max_iter = 0
        offset = self.init_num if self.init_num > 0 else self.budget_list[0]
        for i in range(len(self.budget_list)):
            budget = self.budget_list[i]
            budget_count += budget
            if budget_count > num_train - offset:
                max_iter = i
                break
        else:
            max_iter = len(self.budget_list)

        self.max_iter = max(1, max_iter)

        cfg.ACTIVE_LEARNING.MAX_ITER = self.max_iter
        print(f'Buget list: {self.budget_list}, init index: {self.index} max iter: {self.max_iter}')


    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.max_iter:
            value = self.budget_list[self.index]
            self.index += 1
            return value
        else:
            return None
