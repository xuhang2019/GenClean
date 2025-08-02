import fire
from utils import Config, Logger
from models import Gencleaner
from datasets import load_data

def main(
    run_name=None,
    model_str='raw',
    data_norm='instance',
    use_revin=False,
    debug=False,
):
    # 1. 配置
    config = Config(
        run_name=run_name,
        model_str=model_str,
        data_norm=data_norm,
        use_revin=use_revin,
        debug=debug,
    )
    
    
    logger = Logger(config)

    # 2. 数据
    train_data = load_data(config.train_data_path, subset='train', data_norm=config.data_norm)
    val_data = load_data(config.train_data_path, subset='val', data_norm=config.data_norm)

    # 3. 模型
    gencleaner = Gencleaner(config)

    # 4. 训练
    gencleaner.fit(train_data, val_data, logger)
    logger.close()

if __name__ == '__main__':
    fire.Fire(main)

