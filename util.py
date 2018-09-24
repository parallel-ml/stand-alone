def ssh_client_output(output):
    for host, host_output in output.items():
        for line in host_output.stdout:
            print 'Host [%s] - %s \n' % (host, line)