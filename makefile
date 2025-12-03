models = TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B Llama-2-13b-hf opt-125m opt-1.3b opt-2.7b opt-6.7b opt-13b Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B Qwen2.5-14B llama-7b-hf llama-13b-hf llama-30b-hf

default: ppl

ppl:
	@methods="--eval_clamp --eval_clamp_qwt --wgt_nbit=4 --act_nbit=8"; \
	methods="--eval_base --wgt_nbit=4 --act_nbit=8"; \
	models="TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Qwen2.5-7B"; \
	models="opt-125m"; \
	tasks="wikitext"; \
	start=$$(date +%s); \
	for model in $$models; do \
		for task in $$tasks; do \
			CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py --model_name=$$model --quantize --smooth; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"
scale:
	@methods="--eval_clamp --eval_clamp_qwt --wgt_nbit=4 --act_nbit=8"; \
	methods="--eval_base --wgt_nbit=4 --act_nbit=8"; \
	models="TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Qwen2.5-7B"; \
	models="opt-125m"; \
	tasks="wikitext"; \
	start=$$(date +%s); \
	for model in $$models; do \
		for task in $$tasks; do \
			CUDA_VISIBLE_DEVICES=0 python examples/generate_act_scales.py --model_name=$$model; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"

single:
	model="TinyLlama-1.1B-Chat-v1.0"; \
	CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py --model_name=$$model --n_samples=1
test:
	python test.py