import requests
import os
import io
from PIL import Image, ImageDraw, ImageFont
import time

class MemeGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.pollinations.ai/v1"
        
    def generate_image(self, prompt, output_path="images/meme.jpg"):
        """
        Generate an image using Pollinations.ai API
        
        Args:
            prompt: Text prompt for image generation
            output_path: Path to save the generated image
            
        Returns:
            success: Boolean indicating if the image was generated successfully
        """
        if not self.api_key:
            print("No API key provided. Using fallback image generation.")
            return self.generate_fallback_image(prompt, output_path)
        
        try:
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "prompt": prompt,
                "negative_prompt": "low quality, blurry, text, watermark",
                "width": 512,
                "height": 512,
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
            
            # Make request
            response = requests.post(
                f"{self.base_url}/image/generations",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                # Get image URL
                result = response.json()
                image_url = result.get("output", {}).get("image_url")
                
                if image_url:
                    # Download image
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        # Save image
                        with open(output_path, "wb") as f:
                            f.write(img_response.content)
                        print(f"Image saved to {output_path}")
                        return True
            
            print(f"Failed to generate image: {response.text}")
            return self.generate_fallback_image(prompt, output_path)
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return self.generate_fallback_image(prompt, output_path)
    
    def generate_fallback_image(self, prompt, output_path):
        """Generate a fallback image with text when API fails"""
        try:
            # Create a blank image
            img = Image.new('RGB', (512, 512), color=(50, 50, 50))
            d = ImageDraw.Draw(img)
            
            # Add text
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
                
            # Wrap text
            lines = []
            words = prompt.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if d.textlength(test_line, font=font) <= 480:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw text
            y_position = 50
            for line in lines:
                d.text((20, y_position), line, fill=(255, 255, 255), font=font)
                y_position += 30
            
            # Save the image
            img.save(output_path)
            
            print(f"Fallback image saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating fallback image: {e}")
            return False
    
    def add_text_to_image(self, image_path, text, output_path=None):
        """Add text to an existing image"""
        if output_path is None:
            output_path = image_path
            
        try:
            # Open image
            img = Image.open(image_path)
            d = ImageDraw.Draw(img)
            
            # Add text
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Add semi-transparent background for text
            text_width = d.textlength(text, font=font)
            text_height = 30
            d.rectangle(
                [(10, img.height - text_height - 20), (text_width + 20, img.height - 10)],
                fill=(0, 0, 0, 128)
            )
            
            # Draw text
            d.text((15, img.height - text_height - 15), text, fill=(255, 255, 255), font=font)
            
            # Save image
            img.save(output_path)
            
            print(f"Text added to image and saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error adding text to image: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Initialize generator
    generator = MemeGenerator(api_key="your_api_key_here")
    
    # Generate image
    prompt = "Elon Musk holding a Dogecoin rocket to the moon, digital art"
    generator.generate_image(prompt)
    
    # Add text to image
    generator.add_text_to_image(
        "images/meme.jpg",
        "Elon holding a literal doge rocket to the moon (50k upvotes)"
    )