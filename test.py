import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

from uilib.cfg import WEB_ADDRESS, SPEAKER_DIR, LOGS_DIR, WAVS_DIR, MODEL_DIR, ROOT_DIR
import os
import ChatTTS
CHATTTS_DIR= MODEL_DIR+'/pzc163/chatTTS'
chat = ChatTTS.Chat()
device=os.getenv('device','default')
chat.load(source="custom",custom_path=CHATTTS_DIR, device=None if device=='default' else device,compile=True if os.getenv('compile','true').lower()!='false' else False)



for voice in [1111,1234,2222,2279,3333,4099,4444,4751,5099,5555,6653,6666,7777,7869,8888,9999,1455,1031,125,2328,492,491,1518,1579,1983]:

    torch.manual_seed(voice)
    rand_spk = chat.sample_random_speaker()
    torch.save(rand_spk,f"{SPEAKER_DIR}/{voice}.pt")