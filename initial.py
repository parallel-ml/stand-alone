from multiprocessing import Queue
import yaml
import socket
import os
import ConfigParser

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

HOME = os.environ['HOME']


class Initializer:
    """
        Singleton factory for initializer. The Initializer module has two timers.
        The node_timer is for recording statistics for block1 layer model inference
        time. The timer is for recording the total inference time from last
        fully connected layer.
        Attributes:
            queue: Queue for storing available block1 models devices.
    """
    instance = None

    @classmethod
    def create(cls):
        """ Utilize singleton design pattern to create single instance. """
        if cls.instance is None:
            cls.instance = Initializer()

            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()

            node_config = ConfigParser.ConfigParser()
            node_config.read(HOME + '/node.cfg')
            sys_path = node_config.get('Node Config', 'path', 0)
            sys_split_type = node_config.get('Node Config', 'type', 0)
            node_id = node_config.get('IP Node', ip, 0)

            cls.instance.id = node_id

            # read ip resources from config file
            with open(DIR_PATH + '/resource/conv-config' + sys_path + '/sys-' + sys_split_type + '-config.json') as f:
                configs = yaml.safe_load(f)
                config = configs[node_id]

                for n_id in config['devices']:
                    ip = node_config.get('Node IP', n_id, 0)
                    cls.instance.queue.put(ip)
                cls.instance.split = int(config['split'])
                cls.instance.input_shape = ([int(entry) for entry in config['input_shape'].split(' ')])
        return cls.instance

    def __init__(self):
        self.queue = Queue()
        self.start = 0.0
        self.count = 0
        self.id = ''
        self.split = 0
        self.input_shape = None
