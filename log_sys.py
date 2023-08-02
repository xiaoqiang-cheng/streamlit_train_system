import logging


class XQLOGHandler(object):
    def __new__(cls, *args,**kwargs):
        logger = logging.getLogger('[CYBERXQ]')
        logger.setLevel(logging.DEBUG)
        # 创建文件处理程序
        file_handler = logging.FileHandler('log/model_deploy.log')
        file_handler.setLevel(logging.DEBUG)

        # 创建控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建日志格式器
        formatter = logging.Formatter('%(asctime)s %(name)s  %(levelname)s <%(filename)s:%(lineno)d> $$ %(message)s')

        # 将日志格式器添加到处理程序
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理程序添加到日志记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


if __name__=="__main__":
    test = XQLOGHandler()
    test.info("hjello")