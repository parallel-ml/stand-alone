#!/usr/bin/env bash
# clean
if [ -d "$HOME/stand-alone" ]; then
    rm -rf $HOME/stand-alone
fi

# clone new code, start the system and stop
git clone https://github.com/parallel-ml/stand-alone.git $HOME/stand-alone
