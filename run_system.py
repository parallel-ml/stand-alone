from pssh.pssh_client import ParallelSSHClient
from pssh.utils import load_private_key
from util import ssh_client_output
import argparse
import os
import time
from threading import Thread
from ConfigParser import RawConfigParser

parser = argparse.ArgumentParser('Run layer with customized input and kernal.')
parser.add_argument('channel', type=int, help='Channel size.')
parser.add_argument('kernal', type=int, help='Kernal size.')
parser.add_argument('filter', type=int, help='Filter size.')
parser.add_argument('type', help='Distribution fashion.')
args = parser.parse_args()

channel, kernal, filter, split_type = args.channel, args.kernal, args.filter, args.type

HOME, CUR = os.environ['HOME'], os.environ['PWD']

# python config file parser
config_parser = RawConfigParser()

# find all available hosts
hosts = []
with open("ip", "r") as f:
    for line in f:
        hosts.append(line.rstrip())

config_parser.add_section('Node Config')
config_parser.set('Node Config', 'type', split_type)
config_parser.set('Node Config', 'path', '/{}/{}/{}'.format(kernal, filter, channel))

# map node to ip
node_ip_mapping = dict()
for n in range(1, 6):
    node_ip_mapping.update({'n' + str(n): hosts[n - 1]})

# write ip to node mapping to config file
config_parser.add_section('Node IP')
config_parser.add_section('IP Node')
for node, ip in node_ip_mapping.items():
    config_parser.set('IP Node', ip, node)
    config_parser.set('Node IP', node, ip)
with open('node.cfg', 'wb') as configfile:
    config_parser.write(configfile)
hosts = node_ip_mapping.values()

# ssh commands
pkey = load_private_key(HOME + '/.ssh/id_rsa_pis')
client = ParallelSSHClient(hosts, user='pi', pkey=pkey)

# put node config
client.copy_file('node.cfg', 'node.cfg')


def start_server(ips):
    server_hosts = ParallelSSHClient(ips, user='pi', pkey=pkey)
    servers_outputs = server_hosts.run_command('bash $HOME/stand-alone/scripts/run_system.sh server')
    ssh_client_output(servers_outputs)


def start_client(ips):
    client_host = ParallelSSHClient(ips, user='pi', pkey=pkey)
    client_outputs = client_host.run_command('bash $HOME/stand-alone/scripts/run_system.sh')
    ssh_client_output(client_outputs)


client, servers = hosts[:1], hosts[1:]
for server in servers:
    Thread(target=start_server, args=([server],)).start()
time.sleep(60)
Thread(target=start_client, args=(client,)).start()

