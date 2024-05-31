# 常见问题和报错

1. MacOS 报错 `Initializing libomp.dylib, but found libiomp5.dylib already initialized`

答：在app.py的 import os 的下一行，添加代码

`os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'`


2. MacOS 无报错但进度条一直百分之0 卡住不动

答：app.py中 
`chat.load_models(source="local",local_path=CHATTTS_DIR) `

改为

`chat.load_models(source="local",local_path=CHATTTS_DIR,compile=False)`

3. MacOS 报 `libomp` 相关错误

答：执行 `brew install libomp`

4. 报https相关错误 `ProxyError: HTTPSConnectionPool(host='www.modelscope.cn', port=443)`

答：从 modelscope 魔塔下载模型时不可使用代理，请关闭代理


5. 报错丢失文件 `Missing spk_stat.pt`


答：改成从 modelscope下载模型后，会从魔塔社区下载，但该库里的模型缺少 spk_stat.pt文件，请科学上网后从 https://huggingface.co/2Noise/ChatTTS/blob/main/asset/spk_stat.pt 下载 spk_stat.pt，然后复制 spk_stat.pt 到报错提示的目录下


6. 报错 `Dynamo is not supported on Python 3.12`

答：不支持python3.12+版本，降级到 python3.10

7. MacOS报错 `NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+`

答：执行 	`brew install openssl@1.1` 执行 `pip install urllib3==1.26.15`



8. Windows上报错：`Windows not yet supported for torch.compile`

答：`chat.load_models(compile=False)` 改为 `chat.load_models(compile=False,device="cpu")`

9. Windows上可以运行有GPU，但很慢

答：如果是英伟达显卡，请将cuda升级到11.8+

