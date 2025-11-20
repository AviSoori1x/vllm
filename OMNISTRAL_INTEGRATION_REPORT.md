# Omnistral Model Integration Report

## Overview
This report details the implementation and integration of the **Omnistral** model into the vLLM codebase. Omnistral is an omnimodal model capable of processing text, audio, and image inputs (individually or interleaved) and generating text output.

## Implementation Details

### 1. Model Definition (`vllm/model_executor/models/omnistral.py`)

A new model file was created to house the Omnistral implementation. It composes existing components from **Voxtral** (audio) and **Pixtral** (vision) with a core language model.

#### Key Logic for Robust Mixed-Modality Handling

The core challenge is correctly splitting a flattened list of embeddings back into audio and image groups when they are interleaved. The solution relies on counting `begin_audio` tokens.

**File**: `vllm/model_executor/models/omnistral.py`

```python
class OmnistralForConditionalGeneration(nn.Module, ...):
    # ...
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        # ... (standard handling for None or single modality)

        elif has_audio and has_image:
            # Both modalities - need to split embeddings
            logger.info("Processing mixed audio-image input")

            # CRITICAL: Correctly count audio inputs using begin_audio token
            # This is robust against variable embedding sizes or weird interleaving
            begin_audio_id = self.tokenizer.instruct.audio_encoder.special_ids.begin_audio
            num_audio_inputs = (input_ids == begin_audio_id).sum().item()

            # Split embeddings: Audio comes first in get_multimodal_embeddings
            if num_audio_inputs <= len(multimodal_embeddings):
                audio_embeds = multimodal_embeddings[:num_audio_inputs]
                image_embeds = multimodal_embeddings[num_audio_inputs:]

                # Merge each modality separately
                if audio_embeds:
                    inputs_embeds = self._merge_multimodal_embeddings_safe(
                        input_ids, inputs_embeds, audio_embeds, audio_tok_id)
                if image_embeds:
                    inputs_embeds = self._merge_multimodal_embeddings_safe(
                        input_ids, inputs_embeds, image_embeds, image_tok_id)
            # ... (fallback handling)
```

#### Ensuring Token Consistency

To make the logic above work, we must ensure the processor inserts the `begin_audio` token for every audio input.

**File**: `vllm/model_executor/models/omnistral.py`

```python
class OmnistralMultiModalProcessor(...):
    def _get_prompt_updates(self, ...):
        # ...
        # Audio token replacement logic
        audio_id = processor.audio_token_id
        begin_audio_id = processor.begin_audio_token_id

        def get_audio_replacement(item_idx: int):
            # ... calculate length ...
            # CRITICAL: Must include begin_audio_id so get_input_embeddings can count it
            return [begin_audio_id] + [audio_id] * nb_audio_tokens
        
        # ... return PromptReplacements ...
```

### 2. Registry Integration (`tests/models/registry.py`)

To support testing and recognition by the vLLM system, the model must be registered.

**File**: `tests/models/registry.py`

Add `OmnistralForConditionalGeneration` to the `_MULTIMODAL_EXAMPLE_MODELS` dictionary:

```python
    # ...
    "NVLM_D": _HfExamplesInfo("nvidia/NVLM-D-72B", trust_remote_code=True),
    
    # ADD THIS ENTRY:
    "OmnistralForConditionalGeneration": _HfExamplesInfo(
        "mistralai/Omnistral-Placeholder",
        is_available_online=False,  # Important if weights aren't public yet
    ),
    
    "Llama_Nemotron_Nano_VL": _HfExamplesInfo(
    # ...
```

### 3. Model Executor Registry (`vllm/model_executor/models/registry.py`)

The model class must be mapped to its implementation file so vLLM can load it.

**File**: `vllm/model_executor/models/registry.py`

Add the model to the `_MULTIMODAL_MODELS` dictionary:

```python
_MULTIMODAL_MODELS = {
    # ...
    "NVLM_D": ("nvlm_d", "NVLM_D_Model"),
    
    # ADD THIS ENTRY:
    "OmnistralForConditionalGeneration": (
        "omnistral",
        "OmnistralForConditionalGeneration",
    ),
    
    "Ovis": ("ovis", "Ovis"),
    # ...
}
```

### 4. Configuration Support for Omnimodal Models (`vllm/transformers_utils/configs/mistral.py`)

**CRITICAL FIX**: vLLM had a validation rule preventing vision and audio from being used together. This must be removed for Omnistral to work.

**File**: `vllm/transformers_utils/configs/mistral.py`

#### Remove the Blocking Assertion

**Original code (line ~48):**
```python
assert not (is_vision and is_audio), "Vision and audio are mutually exclusive"

if is_vision:
    config_dict = _remap_mistral_vision_args(config_dict)
if is_audio:
    config_dict = _remap_mistral_audio_args(config_dict)
```

**Updated code:**
```python
# Handle omnimodal (vision + audio) case
if is_vision and is_audio:
    config_dict = _remap_mistral_omnimodal_args(config_dict)
elif is_vision:
    config_dict = _remap_mistral_vision_args(config_dict)
elif is_audio:
    config_dict = _remap_mistral_audio_args(config_dict)
```

#### Add Omnimodal Configuration Function

Add this new function to handle models with both vision and audio:

```python
def _remap_mistral_omnimodal_args(config: dict) -> dict:
    """Remap config for models with both vision and audio (Omnistral)."""
    multimodal_config = config.get("multimodal", {})
    
    # Extract vision config
    vision_config = multimodal_config.get("vision_encoder_args") or config.get("vision_encoder")
    
    # Extract audio config
    whisper_args = multimodal_config.get("whisper_model_args", {})
    encoder_args = whisper_args.get("encoder_args", {})
    downsample_args = whisper_args.get("downsample_args", {})

    quant_config = config.get("quantization_config")
    
    config = {
        "model_type": "omnistral",
        "architectures": ["OmnistralForConditionalGeneration"],
        "text_config": PretrainedConfig.from_dict(config),
        "vision_config": PretrainedConfig.from_dict(vision_config),
        "audio_config": WhisperConfig(
            num_mel_bins=encoder_args["audio_encoding_args"]["num_mel_bins"],
            window_size=encoder_args["audio_encoding_args"]["window_size"],
            sampling_rate=encoder_args["audio_encoding_args"]["sampling_rate"],
            hop_length=encoder_args["audio_encoding_args"]["hop_length"],
            downsample_factor=downsample_args["downsample_factor"],
            d_model=encoder_args["dim"],
            encoder_layers=encoder_args["n_layers"],
            encoder_ffn_dim=encoder_args["hidden_dim"],
            encoder_attention_heads=encoder_args["n_heads"],
            vocab_size=encoder_args["vocab_size"],
            max_source_positions=encoder_args["max_source_positions"],
            is_encoder_decoder=False,
        ),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    
    return config
```

This function:
- Sets `model_type` to `"omnistral"` and `architectures` to `["OmnistralForConditionalGeneration"]`
- Extracts both vision and audio configurations from the model's multimodal config
- Creates a unified config with `text_config`, `vision_config`, and `audio_config`
- Preserves quantization settings if present

**Without this fix, you will get the error**: `"Vision and audio are mutually exclusive"`

## Supported Modalities

The implementation fully supports the following input combinations:

1.  **Text Only**: Standard LLM behavior.
2.  **Audio Only**: Speech-to-text / Audio-conditional generation.
3.  **Image Only**: Visual questioning answering / captioning.
4.  **Mixed/Interleaved**:
    *   `Text + Audio`
    *   `Text + Image`
    *   `Audio + Image`
    *   `Text + Audio + Image` (in any order, with multiple instances of each).

## File Structure Summary

*   `vllm/model_executor/models/omnistral.py`: **New File**. Contains the full model implementation, processor adapter, and dummy input builder.
*   `vllm/model_executor/models/registry.py`: **Modified**. Added model registration in the executor registry.
*   `tests/models/registry.py`: **Modified**. Added registry entry for test support.
*   `vllm/transformers_utils/configs/mistral.py`: **Modified**. Removed the assertion that prevented vision+audio combination and added `_remap_mistral_omnimodal_args` to properly configure Omnistral models.
