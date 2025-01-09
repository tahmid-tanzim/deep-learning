# All Deep Learning related Experiments 

## 1. Three variants of transformer model

### 1.1. Encoder only LLM
- Auto encoding models
- Masked Language Modeling (MLM)
- bi-directional context
- Example: BERT, ROBERTA
- Used for: 
  - Word classification 
  - Token classification
  - Sentiment analysis
  - Named entity recognition

### 1.2. Decoder only LLM
- Auto regressive models
- Casual Language Modeling (CLM)
- unidirectional context
- Example: GPT, BLOOM
- User for:
  - Text generation

### 1.3. Encoder Decoder LLM
- sequence-to-sequence models
- Sentinel token
- Example: T5, BART
- User for:
  - Translation 
  - Summarization
  - Question answering

## 2. Quantization summary
|          | Bits | Exponent  | Fraction | Memory needed to store one value |
|----------|:----:|:---------:|:--------:|----------------------------------|
| FP32     |  32  |     8     |    23    | 4 bytes                          |
| FP16     |  16  |     5     |    10    | 2 bytes                          |
| BFLOAT16 |  16  |     8     |    7     | 2 bytes                          |
| INT8     |  8   |    -/-    |    7     | 1 byte                           |

- Reduced required memory to store and train models.
- Projects original 32-bit floating point numbers into lower precision spaces.
- Quantization-aware training (QAT) learns the quantization scaling factors during training
- BFLOAT16 in a popular choice

## 3. Efficient multi-GPU compute strategies
- Distributed Data Parallel (DDP)
- Fully Sharded Data Parallel (FSDP)