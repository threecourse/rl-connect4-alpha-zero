docker build -t rl-c4 -f dockerfile-rl-c4 . --build-arg USER=$USER --build-arg UID=$(id -u) --build-arg GID=$(id -g)
