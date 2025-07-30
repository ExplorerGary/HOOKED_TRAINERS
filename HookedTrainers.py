# HookedTrainer.py

'''
提供
    HookedTrainer(Trainer)
    HookedSFTTrainer(SFTTrainer)

在第一个训练step时尝试注册DDP钩子，略过第0次初始化模型
'''
try:
    from CustomDDPHooks import allreduce_hook as default_hook
    from CustomDDPHooks import mod_allreduce_hook_base as base_hook
    from CustomDDPHooks import mod_allreduce_hook_EG as EG_hook

except:
    from .CustomDDPHooks import allreduce_hook as default_hook
    from .CustomDDPHooks import mod_allreduce_hook_base as base_hook
    from .CustomDDPHooks import mod_allreduce_hook_EG as EG_hook

import os
import torch.distributed as dist

def dist_check():
    if dist.is_available():
        print(f"Distributed available: ✅")
        if dist.is_initialized():
            print(f"Distributed initialized: ✅ (rank={dist.get_rank()})")
        else:
            print("Distributed available, but not initialized ❌")
    else:
        print("Distributed not available ❌")



# HookedTrainer:


# HookedSFTTrainer:
from trl import SFTTrainer
class HookedSFTTrainer(SFTTrainer):
    def __init__(self, *args, hook_to_register = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_data = []  # Store communication data
        self.hook_registered = False  # Track hook registration
        self.param_name_map = None
        self.checked = False
        self.hook_to_rister = hook_to_register if hook_to_register is not None else default_hook
        
        # 一定有更好的方法解决这个问题
        self.epoch_step_config_0 = None
        self.epoch_step_config_1 = None
        self.output_path = None
        
    def training_step(
        self, model, inputs, num_items_in_batch=None
    ):
        # input args: model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        # --- DDP 钩子 ---
        if not self.checked: # 先检查是否启动了DIST
            dist_check()
            self.checked = True
            
        if self.hook_registered == False: # initializing
            # print(model.module)
            print(f"Hooked??? --- {self.hook_registered}")
            # print(f"dist.is_initiallized --- {dist.is_initialized()}")
            # print(model.type)


        # Make sure allreduce_hook is defined or imported before using it
        # from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
        if dist.is_initialized() and self.hook_registered == False:
            try:
                global param_name_map # 这个是用来了解每一个桶里都装了啥的
                global epoch_step_config_0
                global epoch_step_config_1


                ###### debug info: #######
                try:
                    model_info = f'''
                model.type:
                {model.type}
                ====================================================================================
                model.module.type:
                {model.module.type}
                '''
                    file_path = os.path.join(self.output_path, f"001_model_info_rank_{dist.get_rank()}.txt")
                    with open(file_path, "a") as f:
                        f.write(model_info)
                    print("model structure saved to", file_path)
                except Exception as e:
                    print(f"model structure unable to save...\n{e}")

                
                param_name_map = {id(p): name for name, p in model.named_parameters()}
                self.param_name_map = param_name_map
                
                epoch_step_config_0 =  {"epoch":0,"step":0}   
                self.epoch_step_config_0 = epoch_step_config_0
                
                epoch_step_config_1 = {"epoch":0,"step":0}
                self.epoch_step_config_1 = epoch_step_config_1
                
                print("config initiallized!!!")
                # # Write param_name_map to a CSV file
                # param_map_path = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/param_name_map_rank_{}.csv".format(dist.get_rank())
                # with open(param_map_path, "w", newline="") as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerow(["pid", "name"])
                #     for pid, name in param_name_map.items():
                #         writer.writerow([pid, name])

                # print(list(model.named_parameters()))
                print("registering HOOKS")
                model.register_comm_hook(state=None, hook=self.hook_to_rister)
                self.hook_registered = True
                print("HOOKED!!!")
            except Exception as e:
                print(f"Something bad happened: {e}")



                
        # --- 发现 ---
        # 经过试验，明确 self.model_wrapped才是我们需要处理的东西，用这个注册DDP钩子准备抓取数据！
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print(f"self.model type in training_step: {type(self.model)}")
        #     print(f"self.model_wrapped type in training_step: {type(self.model_wrapped)}") # 已知这个才是我们要找的对象。
        #     # print(self.model == model)
        #     # print(self.model_wrapped == model)
        # 因此，_wrap_model就没必要修改了
        # 但是呢，如果我们需要考虑优化最开始初始化模型到每张GPU上的时候，我们也需要研究新的东西，一个突破口可能是这个_wrap_model (2025年7月30日记录)


        
        # ---调用本家的东西 --- 
        return super().training_step(model,inputs,num_items_in_batch)