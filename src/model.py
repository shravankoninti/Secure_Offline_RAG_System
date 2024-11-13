"""Language model configuration and inference."""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

class ModelManager:
    """
    Manages language model initialization and inference.
    
    Features:
    - 4-bit quantization for efficiency
    - Optimized prompt formatting
    - Configurable generation parameters
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with optimized settings."""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.6,
            top_p=0.90,
            num_beams=2,
            device_map="auto",
            pad_token_id=self.tokenizer.pad_token_id
        )
    
    def format_prompt(self, query: str, context: str) -> str:
        """
        Format prompt with few-shot examples for better response alignment
        """
        prompt = f"""System: You are a helpful AI assistant. Use the provided context to answer the question accurately.
        1. First, try to answer using the provided context with the specified model
        2. Identify patterns and relevant information
        3. If the context doesn't contain the answer, use your internal knowledge to provide an accurate response.
        4. Maintain consistency with the training examples' style and format
        5. If there are any acronym related questions - Strictly, please provide the full form of the answer ONLY. 
        6. If you cannot find the answer in the context by both model and your internal knowledge - Please feel free to hallucinate based on the asked question - as I don't want any question to be unanswered.
        7. Please do not ask me for confirmation on hallucinated answer. Please go ahead and produce the answer but it needs to be very close to query asked.
        8. You need NOT answer the query with saying words like "As per the context" or "According to the provided context" or "Based on the provided context", "As per the provided context" etc., I do not want any reference to sentences like this.Just directly provide the answers.
        8. Think step by step and answer the questions carefully. No need of a big explanation.

    Here are some example questions and answers:

    Question: When should OTA Requestors invoke the ApplyUpdateRequest command?
    Answer: OTA Requestors should invoke the ApplyUpdateRequest command once they are ready to apply a previously downloaded Software Image.

    Answer: The 'WiFiSecurityBitmap' data type encodes the supported Wi-Fi security types present in the Security field of the WiFiInterfaceScanResultStruct.

    Question: What is Matter?
    Answer: Matter is the foundation for connected things, serving as an industry-unifying standard for reliable, seamless, and secure connectivity across smart home devices and ecosystems.

    Question: What is the subject key identifier of the DAC?
    Answer: The subject key identifier of the DAC is 96:C2:D9:24:94:EA:97:85:C0:D1:67:08:E3:88:F1:C0:91:EA:0F:D5.

    Question: acronym of  PDU 
    Answer: Protocol Data Unit

    Now, please answer the following question using the provided context:

    Context: {context}

    Question: {query}

    Answer:"""
        return prompt.strip()
    

    
    def generate_response(self, query: str, context: str) -> str:
        """Get response from the model"""
        try:
            prompt = self.format_prompt(query, context)
            output = self.pipeline(
                prompt,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False
            )
            return output[0]['generated_text'].strip()
        except Exception as e:
            print(f"Error getting model response: {str(e)}")
            return "Error generating response."