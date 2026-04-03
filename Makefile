.PHONY: \
	install-dev \
	install-llm \
	install-tracking \
	build-mcq-synth-pitch-order-trivial \
	build-mcq-synth-loudness-order-trivial \
	build-mcq-synth-duration-order-trivial \
	build-mcq-synth-count-beeps-trivial \
	build-mcq-synth-gap-trivial \
	build-mcq-synth-pattern-pitch-trivial \
	build-mcq-synth-dog-car-order-trivial \
	build-mcq-safety-dataset \
	run-benchmark \
	download-audioflamingo \
	test

DATA_DIR ?= data
RESULTS_DIR ?= results
MCQ_SAFETY_DATASET ?= $(DATA_DIR)/mcq_safety_presence_100.jsonl
MCQ_SYNTH_PITCH_ORDER_TRIVIAL_DATASET ?= $(DATA_DIR)/mcq_synth_pitch_order_trivial_easy.jsonl
MCQ_SYNTH_LOUDNESS_ORDER_TRIVIAL_DATASET ?= $(DATA_DIR)/mcq_synth_loudness_order_trivial_easy.jsonl
MCQ_SYNTH_DURATION_ORDER_TRIVIAL_DATASET ?= $(DATA_DIR)/mcq_synth_duration_order_trivial_easy.jsonl
MCQ_SYNTH_COUNT_BEEPS_TRIVIAL_DATASET ?= $(DATA_DIR)/mcq_synth_count_beeps_trivial_easy.jsonl
MCQ_SYNTH_GAP_TRIVIAL_DATASET ?= $(DATA_DIR)/mcq_synth_gap_trivial_easy.jsonl
MCQ_SYNTH_PATTERN_PITCH_TRIVIAL_DATASET ?= $(DATA_DIR)/mcq_synth_pattern_pitch_trivial_easy.jsonl
MCQ_SYNTH_DOG_CAR_ORDER_TRIVIAL_DATASET ?= $(DATA_DIR)/mcq_synth_dog_car_order_trivial_easy.jsonl
AUDIO_ROOT ?= $(DATA_DIR)/audio

BENCH_TASK ?= mcq-order
BENCH_MODEL ?= random
BENCH_SAMPLES ?= 100
BENCH_RESULTS_ROOT ?= $(RESULTS_DIR)
BENCH_USE_AUDIO ?= 1
BENCH_PREPARE_DATA ?= 1
BENCH_INSTALL_DEPS ?= 0
BENCH_WANDB ?= 1
BENCH_WANDB_PROJECT ?= $(WANDB_PROJECT)
BENCH_WANDB_ENTITY ?= $(WANDB_ENTITY)
BENCH_WANDB_RUN_NAME ?=
BENCH_WANDB_LOG_EVERY ?= $(WANDB_LOG_EVERY)
BENCH_ARGS ?=

BENCH_SAMPLES_ARG := $(if $(BENCH_SAMPLES),--samples $(BENCH_SAMPLES),)
BENCH_USE_AUDIO_ARG := $(if $(filter 1 true yes,$(BENCH_USE_AUDIO)),--use-audio,--disable-audio)
BENCH_PREPARE_DATA_ARG := $(if $(filter 1 true yes,$(BENCH_PREPARE_DATA)),--prepare-data,--no-prepare-data)
BENCH_INSTALL_DEPS_ARG := $(if $(filter 1 true yes,$(BENCH_INSTALL_DEPS)),--install-deps,--no-install-deps)
BENCH_WANDB_ARG := $(if $(filter 1 true yes,$(BENCH_WANDB)),--wandb,--no-wandb)
BENCH_WANDB_ENTITY_ARG := $(if $(strip $(BENCH_WANDB_ENTITY)),--wandb-entity $(BENCH_WANDB_ENTITY),)
BENCH_WANDB_RUN_NAME_ARG := $(if $(strip $(BENCH_WANDB_RUN_NAME)),--wandb-run-name $(BENCH_WANDB_RUN_NAME),)

AF_REPO_URL ?= https://github.com/NVIDIA/audio-flamingo.git
AF_BRANCH ?= audio_flamingo_3
AF_HOME ?= external/audio-flamingo
AF_MODEL_BASE ?= nvidia/audio-flamingo-3
AF_NUM_GPUS ?= 1
AF_BATCH_SIZE ?= 2
AF_MAX_NEW_TOKENS ?= 16
AF_SMOKE_LIMIT ?= 100

QWEN2_AUDIO_MODEL_ID ?= Qwen/Qwen2-Audio-7B-Instruct
QWEN2_AUDIO_BATCH_SIZE ?= 2
QWEN2_AUDIO_MAX_NEW_TOKENS ?= 16
QWEN2_AUDIO_DTYPE ?= float16
QWEN2_AUDIO_DEVICE_MAP ?= auto
QWEN2_AUDIO_SMOKE_LIMIT ?= 100

QWEN2_5_OMNI_MODEL_ID ?= Qwen/Qwen2.5-Omni-7B
QWEN2_5_OMNI_BATCH_SIZE ?= 1
QWEN2_5_OMNI_MAX_NEW_TOKENS ?= 16
QWEN2_5_OMNI_DTYPE ?= float16
QWEN2_5_OMNI_DEVICE_MAP ?= auto
QWEN2_5_OMNI_ATTN ?=
QWEN2_5_OMNI_SMOKE_LIMIT ?= 100
QWEN2_5_OMNI_TRANSFORMERS ?= transformers>=4.57.0
QWEN2_5_OMNI_ATTN_ARG := $(if $(QWEN2_5_OMNI_ATTN),--attn-implementation $(QWEN2_5_OMNI_ATTN),)

VOXTRAL_MODEL_ID ?= mistralai/Voxtral-Mini-3B-2507
VOXTRAL_BATCH_SIZE ?= 2
VOXTRAL_MAX_NEW_TOKENS ?= 16
VOXTRAL_DTYPE ?= float16
VOXTRAL_DEVICE_MAP ?= auto
VOXTRAL_ATTN ?=
VOXTRAL_SMOKE_LIMIT ?= 100
VOXTRAL_TRANSFORMERS ?= transformers>=4.57.0
VOXTRAL_ATTN_ARG := $(if $(VOXTRAL_ATTN),--attn-implementation $(VOXTRAL_ATTN),)

QWEN_MODEL_ID ?= Qwen/Qwen2.5-7B-Instruct
LLAMA_MODEL_ID ?= meta-llama/Llama-3.1-8B-Instruct
LOCAL_DTYPE ?= float16
LOCAL_DEVICE_MAP ?= auto
LOCAL_MAX_NEW_TOKENS ?= 16
LOCAL_TEMPERATURE ?= 0.0
LOCAL_TOP_P ?= 1.0
LOCAL_LIMIT ?= 100
LIMIT_ARG := $(if $(LOCAL_LIMIT),--limit $(LOCAL_LIMIT),)
SYNTH_DIFFICULTY ?= easy
SYNTH_SCENES ?= 500
TRIVIAL_SYNTH_SCENES ?= 100
SYNTH_SEED ?= 7
SYNTH_GENERATOR_VERSION ?= synth-v1

WANDB_PROJECT ?= tacobelal
WANDB_ENTITY ?=
WANDB_LOG_EVERY ?= 50
WANDB_RUN_NAME ?=

WAND_ARGS = --wandb --wandb-project $(WANDB_PROJECT) --wandb-log-every $(WANDB_LOG_EVERY)
ifneq ($(strip $(WANDB_ENTITY)),)
WAND_ARGS += --wandb-entity $(WANDB_ENTITY)
endif
ifneq ($(strip $(WANDB_RUN_NAME)),)
WAND_ARGS += --wandb-run-name $(WANDB_RUN_NAME)
endif

install-llm:
	uv sync --extra llm

install-tracking:
	uv sync --extra tracking

build-mcq-safety-dataset:
	uv run python src/utils/build_safety_check_dataset.py --input $(DATA_DIR)/annotations_strong.csv --output $(MCQ_SAFETY_DATASET)

build-mcq-synth-pitch-order-trivial:
	uv run python src/utils/build_synthetic_mcq_dataset.py \
		--benchmark pitch_order_trivial \
		--difficulty easy \
		--scenes-per-split $(TRIVIAL_SYNTH_SCENES) \
		--seed $(SYNTH_SEED) \
		--audio-root $(AUDIO_ROOT) \
		--dataset-root $(DATA_DIR) \
		--generator-version $(SYNTH_GENERATOR_VERSION)

build-mcq-synth-loudness-order-trivial:
	uv run python src/utils/build_synthetic_mcq_dataset.py \
		--benchmark loudness_order_trivial \
		--difficulty easy \
		--scenes-per-split $(TRIVIAL_SYNTH_SCENES) \
		--seed $(SYNTH_SEED) \
		--audio-root $(AUDIO_ROOT) \
		--dataset-root $(DATA_DIR) \
		--generator-version $(SYNTH_GENERATOR_VERSION)

build-mcq-synth-duration-order-trivial:
	uv run python src/utils/build_synthetic_mcq_dataset.py \
		--benchmark duration_order_trivial \
		--difficulty easy \
		--scenes-per-split $(TRIVIAL_SYNTH_SCENES) \
		--seed $(SYNTH_SEED) \
		--audio-root $(AUDIO_ROOT) \
		--dataset-root $(DATA_DIR) \
		--generator-version $(SYNTH_GENERATOR_VERSION)

build-mcq-synth-count-beeps-trivial:
	uv run python src/utils/build_synthetic_mcq_dataset.py \
		--benchmark count_beeps_trivial \
		--difficulty easy \
		--scenes-per-split $(TRIVIAL_SYNTH_SCENES) \
		--seed $(SYNTH_SEED) \
		--audio-root $(AUDIO_ROOT) \
		--dataset-root $(DATA_DIR) \
		--generator-version $(SYNTH_GENERATOR_VERSION)

build-mcq-synth-gap-trivial:
	uv run python src/utils/build_synthetic_mcq_dataset.py \
		--benchmark gap_trivial \
		--difficulty easy \
		--scenes-per-split $(TRIVIAL_SYNTH_SCENES) \
		--seed $(SYNTH_SEED) \
		--audio-root $(AUDIO_ROOT) \
		--dataset-root $(DATA_DIR) \
		--generator-version $(SYNTH_GENERATOR_VERSION)

build-mcq-synth-pattern-pitch-trivial:
	uv run python src/utils/build_synthetic_mcq_dataset.py \
		--benchmark pattern_pitch_trivial \
		--difficulty easy \
		--scenes-per-split $(TRIVIAL_SYNTH_SCENES) \
		--seed $(SYNTH_SEED) \
		--audio-root $(AUDIO_ROOT) \
		--dataset-root $(DATA_DIR) \
		--generator-version $(SYNTH_GENERATOR_VERSION)

build-mcq-synth-dog-car-order-trivial:
	uv run python src/utils/build_synthetic_mcq_dataset.py \
		--benchmark dog_car_order_trivial \
		--difficulty easy \
		--scenes-per-split $(TRIVIAL_SYNTH_SCENES) \
		--seed $(SYNTH_SEED) \
		--audio-root $(AUDIO_ROOT) \
		--dataset-root $(DATA_DIR) \
		--generator-version $(SYNTH_GENERATOR_VERSION)

run-benchmark:
	uv run python src/utils/run_benchmark.py \
		--task $(BENCH_TASK) \
		--model $(BENCH_MODEL) \
		$(BENCH_SAMPLES_ARG) \
		--results-root $(BENCH_RESULTS_ROOT) \
		$(BENCH_USE_AUDIO_ARG) \
		$(BENCH_PREPARE_DATA_ARG) \
		$(BENCH_INSTALL_DEPS_ARG) \
		$(BENCH_WANDB_ARG) \
		--wandb-project $(BENCH_WANDB_PROJECT) \
		$(BENCH_WANDB_ENTITY_ARG) \
		$(BENCH_WANDB_RUN_NAME_ARG) \
		--wandb-log-every $(BENCH_WANDB_LOG_EVERY) \
		$(BENCH_ARGS)

download-audioflamingo:
	uv run python src/utils/setup_audioflamingo.py --repo-url $(AF_REPO_URL) --branch $(AF_BRANCH) --destination $(AF_HOME)

eval-mcq-order-qwen2-audio-smoke:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order_qwen2_audio.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_AUDIO_MODEL_ID) \
		--batch-size $(QWEN2_AUDIO_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_AUDIO_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_AUDIO_DTYPE) \
		--device-map $(QWEN2_AUDIO_DEVICE_MAP) \
		--limit $(QWEN2_AUDIO_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-audio-full:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order_qwen2_audio.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_AUDIO_MODEL_ID) \
		--batch-size $(QWEN2_AUDIO_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_AUDIO_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_AUDIO_DTYPE) \
		--device-map $(QWEN2_AUDIO_DEVICE_MAP) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-audio-no-audio-smoke:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order_qwen2_audio.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_AUDIO_MODEL_ID) \
		--batch-size $(QWEN2_AUDIO_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_AUDIO_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_AUDIO_DTYPE) \
		--device-map $(QWEN2_AUDIO_DEVICE_MAP) \
		--disable-audio \
		--limit $(QWEN2_AUDIO_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-audio-no-audio-full:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order_qwen2_audio.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_AUDIO_MODEL_ID) \
		--batch-size $(QWEN2_AUDIO_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_AUDIO_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_AUDIO_DTYPE) \
		--device-map $(QWEN2_AUDIO_DEVICE_MAP) \
		--disable-audio \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-5-omni-smoke:
	uv sync --extra tracking
	uv run --with "$(QWEN2_5_OMNI_TRANSFORMERS)" python src/utils/evaluate_mcq_order_qwen2_5_omni.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_5_OMNI_MODEL_ID) \
		--batch-size $(QWEN2_5_OMNI_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_5_OMNI_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_5_OMNI_DTYPE) \
		--device-map $(QWEN2_5_OMNI_DEVICE_MAP) \
		$(QWEN2_5_OMNI_ATTN_ARG) \
		--limit $(QWEN2_5_OMNI_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-5-omni-full:
	uv sync --extra tracking
	uv run --with "$(QWEN2_5_OMNI_TRANSFORMERS)" python src/utils/evaluate_mcq_order_qwen2_5_omni.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_5_OMNI_MODEL_ID) \
		--batch-size $(QWEN2_5_OMNI_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_5_OMNI_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_5_OMNI_DTYPE) \
		--device-map $(QWEN2_5_OMNI_DEVICE_MAP) \
		$(QWEN2_5_OMNI_ATTN_ARG) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-5-omni-no-audio-smoke:
	uv sync --extra tracking
	uv run --with "$(QWEN2_5_OMNI_TRANSFORMERS)" python src/utils/evaluate_mcq_order_qwen2_5_omni.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_5_OMNI_MODEL_ID) \
		--batch-size $(QWEN2_5_OMNI_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_5_OMNI_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_5_OMNI_DTYPE) \
		--device-map $(QWEN2_5_OMNI_DEVICE_MAP) \
		$(QWEN2_5_OMNI_ATTN_ARG) \
		--disable-audio \
		--limit $(QWEN2_5_OMNI_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-5-omni-no-audio-full:
	uv sync --extra tracking
	uv run --with "$(QWEN2_5_OMNI_TRANSFORMERS)" python src/utils/evaluate_mcq_order_qwen2_5_omni.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_5_OMNI_MODEL_ID) \
		--batch-size $(QWEN2_5_OMNI_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_5_OMNI_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_5_OMNI_DTYPE) \
		--device-map $(QWEN2_5_OMNI_DEVICE_MAP) \
		$(QWEN2_5_OMNI_ATTN_ARG) \
		--disable-audio \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-voxtral-smoke:
	uv sync --extra tracking
	uv run --with "$(VOXTRAL_TRANSFORMERS)" python src/utils/evaluate_mcq_order_voxtral.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(VOXTRAL_MODEL_ID) \
		--batch-size $(VOXTRAL_BATCH_SIZE) \
		--max-new-tokens $(VOXTRAL_MAX_NEW_TOKENS) \
		--dtype $(VOXTRAL_DTYPE) \
		--device-map $(VOXTRAL_DEVICE_MAP) \
		$(VOXTRAL_ATTN_ARG) \
		--limit $(VOXTRAL_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-voxtral-full:
	uv sync --extra tracking
	uv run --with "$(VOXTRAL_TRANSFORMERS)" python src/utils/evaluate_mcq_order_voxtral.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(VOXTRAL_MODEL_ID) \
		--batch-size $(VOXTRAL_BATCH_SIZE) \
		--max-new-tokens $(VOXTRAL_MAX_NEW_TOKENS) \
		--dtype $(VOXTRAL_DTYPE) \
		--device-map $(VOXTRAL_DEVICE_MAP) \
		$(VOXTRAL_ATTN_ARG) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-voxtral-no-audio-smoke:
	uv sync --extra tracking
	uv run --with "$(VOXTRAL_TRANSFORMERS)" python src/utils/evaluate_mcq_order_voxtral.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(VOXTRAL_MODEL_ID) \
		--batch-size $(VOXTRAL_BATCH_SIZE) \
		--max-new-tokens $(VOXTRAL_MAX_NEW_TOKENS) \
		--dtype $(VOXTRAL_DTYPE) \
		--device-map $(VOXTRAL_DEVICE_MAP) \
		$(VOXTRAL_ATTN_ARG) \
		--disable-audio \
		--limit $(VOXTRAL_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-voxtral-no-audio-full:
	uv sync --extra tracking
	uv run --with "$(VOXTRAL_TRANSFORMERS)" python src/utils/evaluate_mcq_order_voxtral.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(VOXTRAL_MODEL_ID) \
		--batch-size $(VOXTRAL_BATCH_SIZE) \
		--max-new-tokens $(VOXTRAL_MAX_NEW_TOKENS) \
		--dtype $(VOXTRAL_DTYPE) \
		--device-map $(VOXTRAL_DEVICE_MAP) \
		$(VOXTRAL_ATTN_ARG) \
		--disable-audio \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

test:
	uv sync --extra dev
	uv run pytest
