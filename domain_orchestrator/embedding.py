from transformers import CLIPModel, CLIPProcessor, LlavaForConditionalGeneration, LlavaProcessor
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
import json
import hashlib
from pathlib import Path


#abstract class
class EmbeddingModel:
    """Abstract class for embedding models."""
    @abstractmethod
    def embed_image(self, image_path):
        """Embed a single image."""
        pass


class ClipEmbeddingModel(EmbeddingModel):
    """Handles image and dataset embedding operations."""

    def __init__(self, use_text: bool=False):
        # change this to small backbone for base models
        self.embedding_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to("cuda")
        self.embedding_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.use_text = use_text
        # Initialize the vlm only if text embedding is needed
        if self.use_text:
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.llava_processor = LlavaProcessor.from_pretrained(
                "llava-hf/llava-1.5-7b-hf"
            )
            self.prompt = "Describe what's present in the image in this format: 'A photo of a [object1], [object2], [object3] in the scene' depending on how many major objects are present in the image."
            #self.prompt = "Describe the visual features of the image in only one sentence."
            # self.prompt = "Describe this image in detail. In your description, specifically mention ALL VISIBLE parts of each object in the image. Please, don't generate any blank spaces or new lines in your description."
            #self.prompt = "Describe the total opposite of what is there in the image. Please, don't generate any blank spaces or new lines in your description. Also don't use any preamble like 'this is the total opposite of what is there in the image'. Just describe the total opposite of what is there in the image."
            self.conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]

    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise

        if self.embedding_processor is None or self.embedding_model is None:
            print("Error: CLIP model or processor is not initialized.")
            raise
        
        inputs = self.embedding_processor(images=image, return_tensors="pt").to("cuda")

        # Generate image embeddings
        with torch.no_grad():
            image_embeddings = (
                self.embedding_model.get_image_features(**inputs).detach().cpu().numpy()
            )
        return image_embeddings

    def generate_captions_batch(self, image_paths, batch_size=4) -> list:
        """Generate captions for multiple images in batches."""
        if not self.use_text:
            raise ValueError("Text generation is not enabled. Initialize with use_text=True.")
        
        all_captions = []
        
        # Process images in batches
        from tqdm import tqdm
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating captions in batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load batch of images
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    batch_images.append(image)
                except Exception as e:
                    raise ValueError(f"Error loading image {image_path}: {e}")
            
            # Prepare batch inputs
            prompt_formatted = self.llava_processor.apply_chat_template(self.conversation,
                                                                        tokenize=False,
                                                                        add_generation_prompt=True)
            inputs = self.llava_processor(images=batch_images,
                                          text=[prompt_formatted] * len(batch_images),
                                          return_tensors="pt").to(self.llava_model.device)
            
            print(f"Processing batch of {len(batch_images)} images")
            
            with torch.no_grad():
                outputs = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=100, # check whether it should be 77 or 76, maybe due to different clip tokenizer from llava one??
                    pad_token_id=self.llava_processor.tokenizer.pad_token_id
                )
            
            # Decode captions for valid images
            # check if this input token length logic is correct or not
            #print(f"type and shape of inputs is : {type(inputs)}, {inputs['input_ids'].shape}")
            #print(f"type and shape of outputs is : {type(outputs)}, {outputs.shape}")
            input_token_length = inputs["input_ids"].shape[1]
            batch_captions = []
            
            for j, output in enumerate(outputs):
                generated_tokens = output[input_token_length:]
                generated_text = self.llava_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # Remove multiple spaces between words in generated_text, but why's this happening?
                generated_text = " ".join(generated_text.strip().split())
                if '.' in generated_text:
                    last_period = generated_text.rfind('.')
                    generated_text = generated_text[:last_period+1]
                #print(f"checking if generated text is correct: {generated_text}")
                batch_captions.append(generated_text.strip())
            
            
            all_captions.extend(batch_captions)
            
            print(f"Generated {len(batch_captions)} captions for batch")
        
        return all_captions

    def generate_caption(self, image_path) -> str:
        """Generate a caption for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise
        
        prompt_formatted = self.llava_processor.apply_chat_template(self.conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.llava_processor(images=image, text=prompt_formatted, return_tensors="pt").to("cuda")
        # print(f"type and shape of inputs is : {type(inputs)}")

        with torch.no_grad():
            # 77 is the max number of tokens that clip text encoder can handle
            #import time
            #start_time = time.time()
            # this takes 7.5 seconds now for a single image
            outputs = (
                self.llava_model.generate(**inputs, max_new_tokens=77,
                pad_token_id = self.llava_processor.tokenizer.pad_token_id)
            )
            #print(f"Time taken to generate outputs from the llava model: {time.time() - start_time} seconds")

        input_token_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_token_length:]
        generated_text = self.llava_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        #generated_text = generated_text.replace(self.prompt, "").strip()
        print(f"Generated caption for image {image_path}: {generated_text.strip()}")

        return generated_text

    def embed_text(self, text) -> npt.NDArray:
        """Embed a single text."""
        inputs = self.embedding_processor(text=text, return_tensors="pt", truncation=True).to("cuda")
        input_token_count = inputs["input_ids"].shape[1]
        print(f"Text to be fed into the clip text encoder: {text}")
        print(f"Input token count for the caption to be fed into the clip text encoder: {input_token_count}")
        
        with torch.no_grad():
            text_embeddings = (
                self.embedding_model.get_text_features(**inputs).detach().cpu().numpy()
            )

        return text_embeddings


class EmbeddingManager:
    """Handles image and dataset embedding operations."""
    
    def __init__(self, use_text: bool=False, image_weight: float=0.5, text_weight: float=0.5):
        self.embedding_model = ClipEmbeddingModel(use_text=use_text)
        self.use_text = use_text
        
        # Weights for combining image and text embeddings
        self.image_weight = image_weight
        self.text_weight = text_weight
        
        # Normalize weights to ensure they sum to 1
        total_weight = image_weight + text_weight
        self.image_weight = image_weight / total_weight
        self.text_weight = text_weight / total_weight
        
        if self.use_text:
            # this is misleading when using "combination_weights"
            # print(f"Using embedding weights - Image: {self.image_weight:.2f}, Text: {self.text_weight:.2f}")
            pass
        else:
            print("Using image embeddings only")
    
    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        return self.embedding_model.embed_image(image_path)

    def embed_text_for_image(self, image_path) -> npt.NDArray:
        """Generate caption for an image and embed the caption."""
        import time
        start_time = time.time()
        caption = self.embedding_model.generate_captions_batch([image_path], batch_size=1)[0]
        print(f"Time taken to generate caption: {time.time() - start_time} seconds")

        return self.embedding_model.embed_text(caption)
    
    def embed_dataset(self, dataset_path, debug=False) -> npt.NDArray:
        """Embed all images in a dataset."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")

        print(f"Embedding dataset from '{dataset_path}' ...")
        dataset_embeddings = []
        dataset_text_embeddings = []
        dataset_mixed_embeddings = []

        # IDD has both png and jpg images in train set
        image_files = list(dataset_path.rglob("*.png")) + list(dataset_path.rglob("*.jpg"))
    
        if not image_files:
            print(f"Warning: No images found in dataset path '{dataset_path}'.")
            return []

        from tqdm import tqdm
        
        # First, generate all image embeddings
        for img in tqdm(image_files, desc="Generating image embeddings"):
            embedding = self.embed_image(img)
            # print(f"Shape and type of image embedding is: {embedding.shape}, {type(embedding)}")
            if embedding is not None:
                dataset_embeddings.append(embedding)
            else:
                raise ValueError(f"Error embedding image '{img}'.")
        
        # Then, generate all text embeddings in batches if use_text is enabled
        if self.use_text:
            print("Generating text embeddings in batches...")
            captions = self.embedding_model.generate_captions_batch(image_files, batch_size=128)
            
            # Calculate and print cosine similarities
            cosine_similarities = []
            
            for i, (img, caption) in enumerate(tqdm(zip(image_files, captions), desc="Generating text embeddings", total=len(image_files))):
                if caption:  # Only process if caption was generated successfully
                    caption_embedding = self.embedding_model.embed_text(caption)
                    #print(f"Shape and type of text embedding is : {caption_embedding.shape}, {type(caption_embedding)}")
                    if caption_embedding is not None:
                        dataset_text_embeddings.append(caption_embedding)
                        
                        # Calculate cosine similarity between image and text embeddings
                        image_emb = dataset_embeddings[i]
                        text_emb = caption_embedding
                        
                        # Normalize embeddings for cosine similarity
                        image_emb_norm = image_emb / np.linalg.norm(image_emb)
                        text_emb_norm = text_emb / np.linalg.norm(text_emb)
                        
                        # Calculate cosine similarity
                        cosine_sim = np.dot(image_emb_norm.flatten(), text_emb_norm.flatten())
                        cosine_similarities.append(cosine_sim)

                        text_along_image = cosine_sim * text_emb
                        text_along_image_norm = np.linalg.norm(text_along_image)
                        text_orthogonal_to_image = text_emb - text_along_image
                        text_orthogonal_to_image_norm = np.linalg.norm(text_orthogonal_to_image)
                        
                        # Print similarity for first few images and some random samples
                        #if i < 5 or i % 100 == 0:
                        print(f"Image {i}: {img.name}")
                        print(f"  Caption: {caption}")
                        print(f"  Cosine similarity: {cosine_sim:.4f}")
                        print(f"  Text along image: {text_along_image_norm:.4f}")
                        print(f"  Text orthogonal to image: {text_orthogonal_to_image_norm:.4f}")
                        
                        # calculate mixed embeddings (changed to text_along_image)
                        mixed_embedding = 1 * dataset_embeddings[i] + self.text_weight * text_orthogonal_to_image
                        dataset_mixed_embeddings.append(mixed_embedding)
                    else:
                        raise ValueError(f"Error embedding text for image '{img}'.")
                else:
                    # If caption generation failed, raise error
                    raise ValueError(f"Error generating caption for image '{img}'.")


        print("Finished embedding dataset.")

        return dataset_embeddings, dataset_text_embeddings, dataset_mixed_embeddings

    # change this to handle naming conventions when text embeddings are also enabled
    def calculate_statistics(self, domain_name, domain_path, train_path, image_weight=None, text_weight=None, force_embedding=False):
        """
        Calculate or load domain statistics.
        Args:
            domain_name (str): The name of the domain.
            domain_path (Path): The path to the domain database where the statistics will be saved.
            train_path (Path): The path to the train set.
            image_weight (float): Weight for image embeddings (used for file naming).
            text_weight (float): Weight for text embeddings (used for file naming).
            force_embedding (bool): If True, regenerate embeddings even if they already exist.
        Returns:
            dict: A dictionary containing the statistics.
        """
        # Determine suffix based on whether weights are provided
        if image_weight is not None and text_weight is not None:
            print("USING text weihgted embeddings")
            suffix = f"_{image_weight:.1f}_{text_weight:.1f}_statistics.npz"
        else:
            print("USING THE ORIGINAL SEMLA EMBEDDINGS")
            suffix = "_statistics.npz"
            
        statistics_path = domain_path / f"{domain_name}{suffix}"
        stats_dict = {}

        print(f"Statistics file: {statistics_path}")
        
        # Skip loading existing file if force_embedding is True
        if not force_embedding and statistics_path.exists():  # Load the data if it exists
            try:
                print(f"Loading statistics from {domain_name}{suffix} ...")
                stats = np.load(statistics_path)
                stats_dict.update({
                    "train_average_embedding": stats["train_average_embedding"],
                })
                
                print(f"Statistics loaded from {domain_name}{suffix}")
                return stats_dict
            except Exception as e:
                print(f"Error loading statistics file '{statistics_path}': {e}")
                return None

        if force_embedding:
            print(f"Force embedding enabled - regenerating statistics for domain '{domain_name}' ...")
        else:
            print(f"Statistics file {statistics_path} does not exist, calculating statistics for domain '{domain_name}' ...")
            
        train_dataset_embeddings, train_dataset_text_embeddings, train_dataset_mixed_embeddings = self.embed_dataset(train_path)

        if not train_dataset_embeddings:
            raise ValueError("No embeddings were generated for dataset.")

        try:
            # if using text, calculate mixed embeddings
            if self.use_text:
                train_average_embedding = np.mean(train_dataset_mixed_embeddings, axis=0)
            else:
                train_average_embedding = np.mean(train_dataset_embeddings, axis=0)
        except Exception as e:
            print(f"Error computing mean embedding: {e}")
            raise

        stats_dict.update({
            "train_average_embedding": train_average_embedding
        })


        try:
            np.savez(
                statistics_path,
                train_average_embedding=train_average_embedding,
            )
            print(f"Statistics saved to {domain_name}{suffix}")
        except Exception as e:
            print(f"Error saving statistics file '{statistics_path}': {e}")
            raise
        return stats_dict

    def calculate_statistics_multiple_weights(self, domain_name, domain_path, train_path, weight_combinations, force_embedding=False):
        """
        Calculate domain statistics for multiple weight combinations efficiently.
        Args:
            domain_name (str): The name of the domain.
            domain_path (Path): The path to the domain database where the statistics will be saved.
            train_path (Path): The path to the train set.
            weight_combinations (list): List of (image_weight, text_weight) tuples.
            force_embedding (bool): If True, regenerate embeddings even if they already exist.
        """
        print(f"Processing {len(weight_combinations)} weight combinations for domain '{domain_name}'")
        
        # Check if any statistics files exist and force_embedding is False
        existing_files = []
        if not force_embedding:
            for img_w, txt_w in weight_combinations:
                suffix = f"_{img_w:.1f}_{txt_w:.1f}_statistics.npz"
                statistics_path = domain_path / f"{domain_name}{suffix}"
                if statistics_path.exists():
                    existing_files.append((img_w, txt_w))
        
        if existing_files and not force_embedding:
            print(f"Found existing files for {len(existing_files)} weight combinations, skipping...")
            for img_w, txt_w in existing_files:
                print(f"  - Image: {img_w}, Text: {txt_w}")
        
        # Get weight combinations that need to be processed
        combinations_to_process = [combo for combo in weight_combinations if combo not in existing_files] if not force_embedding else weight_combinations
        
        if not combinations_to_process:
            print("All weight combinations already exist, skipping processing.")
            return
        
        print(f"Processing {len(combinations_to_process)} weight combinations...")
        
        # Calculate embeddings once (image and text if use_text is enabled)
        print("Calculating base embeddings (image and text)...")
        train_dataset_embeddings, train_dataset_text_embeddings, _ = self.embed_dataset(train_path)
        
        if not train_dataset_embeddings:
            raise ValueError("No embeddings were generated for dataset.")
        
        # Process each weight combination
        for img_w, txt_w in combinations_to_process:
            print(f"Processing weights - Image: {img_w}, Text: {txt_w}")
            
            # Normalize weights
            total_weight = img_w + txt_w
            normalized_img_w = img_w / total_weight
            normalized_txt_w = txt_w / total_weight
            
            # Calculate weighted embeddings
            if self.use_text:
                train_dataset_mixed_embeddings = []
                for i in range(len(train_dataset_embeddings)):
                    mixed_embedding = normalized_img_w * train_dataset_embeddings[i] + normalized_txt_w * train_dataset_text_embeddings[i]
                    train_dataset_mixed_embeddings.append(mixed_embedding)
                train_average_embedding = np.mean(train_dataset_mixed_embeddings, axis=0)
            else:
                train_average_embedding = np.mean(train_dataset_embeddings, axis=0)
            
            # Save statistics
            suffix = f"_{img_w:.1f}_{txt_w:.1f}_statistics.npz"
            statistics_path = domain_path / f"{domain_name}{suffix}"
            
            try:
                np.savez(
                    statistics_path,
                    train_average_embedding=train_average_embedding,
                )
                print(f"Statistics saved to {domain_name}{suffix}")
            except Exception as e:
                print(f"Error saving statistics file '{statistics_path}': {e}")
                raise
        
        print(f"Completed processing {len(combinations_to_process)} weight combinations for domain '{domain_name}'")

    def get_weighted_embedding_for_image(self, image_path) -> npt.NDArray:
        """Get weighted embedding for a single image (image + text if use_text is enabled)."""
        # Get image embedding
        image_embedding = self.embed_image(image_path)
        
        if self.use_text:
            # Generate caption and get text embedding
            text_embedding = self.embed_text_for_image(image_path)
            
            # Calculate weighted sum
            weighted_embedding = self.image_weight * image_embedding + self.text_weight * text_embedding
            # print(f"Weighted embedding for image {image_path}: {weighted_embedding}")

            return weighted_embedding
        else:
            # Return only image embedding if text is not enabled
            return image_embedding

    def get_weighted_embeddings_batch(self, image_paths, batch_size=64) -> list:
        """Get weighted embeddings for multiple images in batches."""
        all_weighted_embeddings = []
        
        # Process images in batches
        from tqdm import tqdm
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating weighted embeddings in batches"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Get image embeddings using the same method as single-image processing
            batch_image_embeddings = []
            for image_path in batch_paths:
                image_embedding = self.embed_image(image_path)
                batch_image_embeddings.append(image_embedding)
            
            if self.use_text:
                # Generate captions for the batch
                batch_captions = self.embedding_model.generate_captions_batch(batch_paths, batch_size=len(batch_paths))
                
                # Get text embeddings using the same method as single-image processing
                batch_text_embeddings = []
                for caption in batch_captions:
                    text_embedding = self.embedding_model.embed_text(caption)
                    batch_text_embeddings.append(text_embedding)
                
                # Calculate weighted embeddings for the batch
                # make changes to calculate weighted embeddings in directed way
                for img_emb, txt_emb in zip(batch_image_embeddings, batch_text_embeddings):
                    image_emb_norm = img_emb / np.linalg.norm(img_emb)
                    text_emb_norm = txt_emb / np.linalg.norm(txt_emb)
                    cosine_sim = np.dot(image_emb_norm.flatten(), text_emb_norm.flatten())
                    text_along_image = cosine_sim * txt_emb
                    text_orthogonal_to_image = txt_emb - text_along_image
                    weighted_embedding = 1 * img_emb + self.text_weight * text_orthogonal_to_image
                    all_weighted_embeddings.append(weighted_embedding)
            else:
                # Return only image embeddings if text is not enabled
                all_weighted_embeddings.extend(batch_image_embeddings)
        
        return all_weighted_embeddings
