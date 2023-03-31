# CCEval评测集的baseline
# 此baseline采用m2m-418M模型进行zh->{vi,mn,lo}和{vi,mn,lo}->zh共6个方向的评测
# 评测采用sacrebleu进行计算
# 为安全起见，本版本暂时未提供任何ground-truth。后需确认后会进一步开放下载

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import numpy as np
import torch
import math
from tqdm import tqdm
import sacrebleu
import json
device = torch.device('cuda:0')

class M2M():
	def __init__(self):
		self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
		self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
		print("NMT model loading finished !")

	def predict(self, text, src_lang, tgt_lang):
		self.tokenizer.src_lang = src_lang
		encoded_zh = self.tokenizer(text, return_tensors="pt", padding=True)
		self.model.to(device)
		encoded_zh.to(device)
		generated_tokens = self.model.generate(**encoded_zh, forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang))
		output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
		return output

	def evaluate(self, srcs, refs, src_lang, tgt_lang):
		lines = srcs
		batch_size = 32
		batch_num = math.ceil(len(lines)/batch_size)
		hyps = []
		for i in tqdm(range(batch_num)):
			inputs = lines[i*batch_size: (i+1)*batch_size]
			outputs = self.predict(inputs, src_lang, tgt_lang)
			hyps.extend(outputs)
		assert len(refs) == len(hyps)

		tokenize = "13a" if tgt_lang!="zh" else "zh"
		score = sacrebleu.corpus_bleu(hyps, [refs], tokenize = tokenize).score
		return round(score,4)

def main():
	score = {}
	# 评测zh->{vi,mn,lo}
	data= {"zh":[], "vi":[], "lo":[], "mn":[]}
	with open("CCEval_valid.json","r") as r:
		for line in r:
			line = json.loads(line)
			data[line["src_lang"]].append(line["source"])

	m2m = M2M()

	src_lang = "zh"
	for tgt_lang in "vi,lo,mn".split(","):
		srcs = data["zh"]
		score[f"{src_lang}2{tgt_lang}"] = m2m.evaluate(srcs, refs, src_lang, tgt_lang)

	tgt_lang = "zh"
	for src_lang in "vi,lo,mn".split(","):
		srcs = data[src_lang]
		score[f"{src_lang}2{tgt_lang}"] = m2m.evaluate(srcs, refs, src_lang, tgt_lang)

if __name__=="__main__":
	main()
