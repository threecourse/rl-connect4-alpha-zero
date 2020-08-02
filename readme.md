## Connect Fourのアルファ碁ゼロメソッドによる強化学習

以下ブログ記事も参考にして下さい：  
https://threecourse.hatenablog.com/entry/2020/08/02/115804

### docker

* nvidia-docker2 を利用

docker imageのビルド

```
cd docker 
./docker.sh
```

コンテナの起動

```
# ユーザとしてdockerに入るようにしている
# （参考）https://qiita.com/tnarihi/items/275c009e9dec1306893f
export DOCKER_USER="--user=$(id -u):$(id -g) $(for i in $(id -G); do echo -n "--group-add "$i " "; done) --volume=/etc/group:/etc/group:ro --volume=/etc/passwd:/etc/passwd:ro --volume=/etc/shadow:/etc/shadow:ro --volume=/etc/sudoers.d:/etc/sudoers.d:ro"
export DOCKER_PORT="-p 8888:8888 -p 6006-6015:6006-6015"
export DOCKER_DISP="$DOCKER_USER $DOCKER_PORT --env=DISPLAY=$DISPLAY --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw -e QT_X11_NO_MITSHM=1"
docker run -it --rm --runtime=nvidia $DOCKER_DISP -v $PWD:/tmp/work -w /tmp/work --name rl-c4 rl-c4
```

起動中のコンテナに入る
```
docker exec -it rl-c4 bash
```

### 学習

起動したコンテナ内で、以下を同時に実行する  
```
# 自己対戦
python src/c4_zero/run.py self -c config/c4.yml
# 学習（メモリリーク防止のため、一定時間ごとに再起動する仕組みとしている）
python src/c4_zero/run.py opt-workaround -c config/c4.yml
# tensorboardの表示（学習には必須ではない）
tensorboard --logdir logs/tensorboard/
```

その他のコマンド
```
# GUIとの対戦を行う場合
mv data-trained data  # 学習済みデータを利用するようにする
python src/c4_zero/run.py play_gui -c config/c4.yml
```

### ライセンス

* MITライセンス
  * https://github.com/mokemokechicken/reversi-alpha-zero (MITライセンス）をforkしたものを元にしています  
  * ソルバー`src/c4_zero/env/solver/solver-pascal`のライセンスはAGPLのため、この部分を使用する場合は全体のライセンスにもご注意下さい      
