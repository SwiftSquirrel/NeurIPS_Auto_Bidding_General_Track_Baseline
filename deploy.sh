# docker build -t registry.cn-beijing.aliyuncs.com/nips_rtb_bjwd/rtb_bjwd:0.1 -f ./Dockerfile .
# docker login --username=海边的蜗牛 registry.cn-beijing.aliyuncs.com
docker build -f ./Dockerfile -t registry.cn-beijing.aliyuncs.com/nips_rtb_bjwd/rtb_bjwd:0.1 .

docker push registry.cn-beijing.aliyuncs.com/nips_rtb_bjwd/rtb_bjwd:0.1

