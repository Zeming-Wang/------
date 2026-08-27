import argparse
from maas.configs.models_config import ModelsConfig
from maas.ext.maas.scripts.optimizer import Optimizer
from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS

def parse_args():
    parser = argparse.ArgumentParser(description="MAAS Optimizer")
    parser.add_argument(
        #定义了数据集来源
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset type",
    )
    parser.add_argument("--sample", type=int, default=4, help="Sample count")
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="maas/ext/maas/scripts/optimized",
        help="Optimized result save path",
    )
    
    #定义了一些超参数和模型
    parser.add_argument("--round", type=int, default=1, help="choice the round of optimized")
    parser.add_argument("--batch_size", type=int, default=4, help="Train batch size")
    parser.add_argument(
        "--opt_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Specifies the name of the model used for optimization tasks.",
    )
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Specifies the name of the model used for execution tasks.",
    )
    
    #算法开关
    
    parser.add_argument("--is_test",type=bool, default=False, help="choice the optimizer mode")
    parser.add_argument("--is_textgrad", type = bool, default=False, help="choice to use textgrad")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = EXPERIMENT_CONFIGS[args.dataset] #获取对应数据集和算子库和问题类型

    models_config = ModelsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{args.opt_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --opt_model_name flag. "
        )

    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --exec_model_name flag. "
        )

    optimizer = Optimizer(
        dataset=config.dataset, 
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        round=args.round,
        batch_size=args.batch_size,
        lr=args.lr,
        is_textgrad=args.is_textgrad,
    )
    #将相关的超参数传入到optimizer之中

    if args.is_test:
        optimizer.optimize("Test")      
    else:
        optimizer.optimize("Graph")
