#!/usr/bin/env bash
# start server and waits a few time
# if argument provided, then start the server
if [[ -n $1 ]]; then
    cd $HOME/stand-alone && python server.py
else
    cd $HOME/stand-alone && python client.py
fi
