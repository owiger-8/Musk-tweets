import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import threading
import time
import json
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageDraw, ImageFont
import io
import yfinance as yf
from datetime import datetime, timedelta
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)
 
CNN_MODEL = None
REDDIT_MODEL = None
GOLDMAN_MODEL = None
MARKET_MODEL = None

class MuskTweetsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MUSK-TWEETS")
        self.root.geometry("1200x800")
        self.root.configure(bg="black")
        
        # Set up the main frame
        self.main_frame = ctk.CTkFrame(self.root, fg_color="black")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize UI components
        self.setup_initial_ui()
        
        # Load models in background
        self.load_models_thread = threading.Thread(target=self.load_models)
        self.load_models_thread.daemon = True
        self.load_models_thread.start()
        
    def setup_initial_ui(self):
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="MUSK-TWEETS", 
            font=("Arial", 60, "bold"),
            text_color="white"
        )
        self.title_label.pack(pady=(100, 50))
        
        # Input field
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="#555555", corner_radius=30)
        self.input_frame.pack(pady=20, padx=100, fill=tk.X)
        
        self.tweet_input = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="What's today's Tweet ????",
            font=("Arial", 18),
            fg_color="#555555",
            text_color="white",
            border_width=0
        )
        self.tweet_input.pack(pady=15, padx=20, fill=tk.X)
        
        # Analyze button
        self.analyze_button = ctk.CTkButton(
            self.main_frame,
            text="Analyze",
            font=("Arial", 18, "bold"),
            fg_color="#FF3333",
            hover_color="#CC0000",
            corner_radius=30,
            command=self.start_analysis
        )
        self.analyze_button.pack(pady=30)
        
    def setup_loading_ui(self):
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Tweet display
        self.tweet_display_frame = ctk.CTkFrame(self.main_frame, fg_color="#555555", corner_radius=30)
        self.tweet_display_frame.pack(pady=(20, 30), padx=100, fill=tk.X)
        
        self.tweet_display = ctk.CTkLabel(
            self.tweet_display_frame,
            text=f"Elon: {self.tweet_text}",
            font=("Arial", 18),
            text_color="white"
        )
        self.tweet_display.pack(pady=15, padx=20)
        
        # Create grid for loading elements
        self.grid_frame = ctk.CTkFrame(self.main_frame, fg_color="black")
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # CNN News loading (top left)
        self.cnn_frame = ctk.CTkFrame(self.grid_frame, fg_color="#FF0000", corner_radius=20)
        self.cnn_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)
        
        self.cnn_logo = ctk.CTkLabel(
            self.cnn_frame,
            text="CNN",
            font=("Arial", 30, "bold"),
            text_color="white"
        )
        self.cnn_logo.pack(pady=20)
        
        self.cnn_loading = self.create_loading_spinner(self.cnn_frame)
        self.cnn_loading.pack(pady=20)
        
        # Reddit Meme loading (bottom left)
        self.reddit_frame = ctk.CTkFrame(self.grid_frame, fg_color="#555555", corner_radius=20)
        self.reddit_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.reddit_loading = self.create_loading_spinner(self.reddit_frame)
        self.reddit_loading.pack(pady=40)
        
        # Goldman Sachs loading (bottom right)
        self.goldman_frame = ctk.CTkFrame(self.grid_frame, fg_color="#555555", corner_radius=20)
        self.goldman_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.goldman_loading = self.create_loading_spinner(self.goldman_frame)
        self.goldman_loading.pack(pady=40)
        
        # Market prediction loading (right)
        self.market_frame = ctk.CTkFrame(self.grid_frame, fg_color="#555555", corner_radius=20)
        self.market_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.market_loading = self.create_loading_spinner(self.market_frame)
        self.market_loading.pack(pady=100)
        

        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)
        self.grid_frame.grid_columnconfigure(2, weight=2)
        self.grid_frame.grid_rowconfigure(0, weight=1)
        self.grid_frame.grid_rowconfigure(1, weight=1)
        
    def create_loading_spinner(self, parent):
        # Create a simple loading spinner using a label with animated dots
        loading_label = ctk.CTkLabel(
            parent,
            text="",
            font=("Arial", 24),
            text_color="white"
        )
        
        def animate_dots():
            dots = ["⚪⚪⚪⚪⚪⚪⚪⚪", "⚫⚪⚪⚪⚪⚪⚪⚪", "⚫⚫⚪⚪⚪⚪⚪⚪", 
                    "⚫⚫⚫⚪⚪⚪⚪⚪", "⚫⚫⚫⚫⚪⚪⚪⚪", "⚫⚫⚫⚫⚫⚪⚪⚪",
                    "⚫⚫⚫⚫⚫⚫⚪⚪", "⚫⚫⚫⚫⚫⚫⚫⚪", "⚫⚫⚫⚫⚫⚫⚫⚫",
                    "⚪⚫⚫⚫⚫⚫⚫⚫", "⚪⚪⚫⚫⚫⚫⚫⚫", "⚪⚪⚪⚫⚫⚫⚫⚫",
                    "⚪⚪⚪⚪⚫⚫⚫⚫", "⚪⚪⚪⚪⚪⚫⚫⚫", "⚪⚪⚪⚪⚪⚪⚫⚫",
                    "⚪⚪⚪⚪⚪⚪⚪⚫"]
            i = 0
            
            def update_dots():
                nonlocal i
                if hasattr(loading_label, 'is_active') and loading_label.is_active:
                    loading_label.configure(text=dots[i])
                    i = (i + 1) % len(dots)
                    loading_label.after(100, update_dots)
            
            loading_label.is_active = True
            update_dots()
            
        animate_dots()
        return loading_label
        
    def setup_results_ui(self, results):
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Tweet display
        self.tweet_display_frame = ctk.CTkFrame(self.main_frame, fg_color="#555555", corner_radius=30)
        self.tweet_display_frame.pack(pady=(20, 30), padx=100, fill=tk.X)
        
        self.tweet_display = ctk.CTkLabel(
            self.tweet_display_frame,
            text=f"Elon: {self.tweet_text}",
            font=("Arial", 18),
            text_color="white"
        )
        self.tweet_display.pack(pady=15, padx=20)
        
        # Create grid for result elements
        self.grid_frame = ctk.CTkFrame(self.main_frame, fg_color="black")
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # CNN News (top left)
        self.cnn_frame = ctk.CTkFrame(self.grid_frame, fg_color="#FF0000", corner_radius=20)
        self.cnn_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)
        
        self.cnn_logo = ctk.CTkLabel(
            self.cnn_frame,
            text="CNN",
            font=("Arial", 30, "bold"),
            text_color="white"
        )
        self.cnn_logo.pack(pady=(20, 10))
        
        self.cnn_headline = ctk.CTkLabel(
            self.cnn_frame,
            text=results["title"].upper(),
            font=("Arial", 20, "bold"),
            text_color="white",
            wraplength=500
        )
        self.cnn_headline.pack(pady=(0, 20), padx=20)
        
        # Reddit Meme (bottom left)
        self.reddit_frame = ctk.CTkFrame(self.grid_frame, fg_color="#555555", corner_radius=20)
        self.reddit_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Load and display the meme image
        if os.path.exists("images/meme.jpg"):
            img = Image.open("images/meme.jpg")
            img = img.resize((250, 250), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.meme_image_label = tk.Label(self.reddit_frame, image=photo, bg="#555555")
            self.meme_image_label.image = photo  # Keep a reference
            self.meme_image_label.pack(pady=10)
            
            self.meme_title = ctk.CTkLabel(
                self.reddit_frame,
                text=results["meme"],
                font=("Arial", 12),
                text_color="white",
                wraplength=250
            )
            self.meme_title.pack(pady=5)
        
        # Goldman Sachs (bottom right)
        self.goldman_frame = ctk.CTkFrame(self.grid_frame, fg_color="#555555", corner_radius=20)
        self.goldman_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        # Display Goldman Sachs logo or image
        self.goldman_logo = ctk.CTkLabel(
            self.goldman_frame,
            text="Handelsblatt",
            font=("Arial", 20, "bold"),
            text_color="white"
        )
        self.goldman_logo.pack(pady=(20, 10))
        
        self.goldman_report = ctk.CTkLabel(
            self.goldman_frame,
            text=results["report"],
            font=("Arial", 14),
            text_color="white",
            wraplength=250
        )
        self.goldman_report.pack(pady=10, padx=20)
        
        # Market prediction (right)
        self.market_frame = ctk.CTkFrame(self.grid_frame, fg_color="white", corner_radius=20)
        self.market_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        # Create the stock chart
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Get current date
        current_date = datetime.now()
        dates = [(current_date + timedelta(days=i)).strftime('%d %b') for i in range(6)]
        
        # Combine current price with predicted prices
        all_prices = [results["price"]] + results["output"]
        
        # Plot the data
        ax.plot(dates, all_prices, 'r-', linewidth=2)
        ax.fill_between(dates, all_prices, color='red', alpha=0.2)
        
        # Add labels and grid
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create tabs for different time periods
        time_periods = ["1D", "5D", "1M", "6M", "YTD", "1Y"]
        tab_frame = ctk.CTkFrame(self.market_frame, fg_color="white")
        tab_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        for i, period in enumerate(time_periods):
            color = "#0066FF" if period == "5D" else "#888888"
            tab = ctk.CTkLabel(
                tab_frame,
                text=period,
                font=("Arial", 14, "bold"),
                text_color=color
            )
            tab.pack(side=tk.LEFT, padx=15)
        
        # Add the plot to the UI
        canvas = FigureCanvasTkAgg(fig, master=self.market_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Display the current stock price
        self.price_label = ctk.CTkLabel(
            self.market_frame,
            text=f"${results['price']:.2f}",
            font=("Arial", 50, "bold"),
            text_color="#FF3333"
        )
        self.price_label.pack(pady=(0, 20))
        
        # Configure grid weights
        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)
        self.grid_frame.grid_columnconfigure(2, weight=2)
        self.grid_frame.grid_rowconfigure(0, weight=1)
        self.grid_frame.grid_rowconfigure(1, weight=1)
        
        # Add a restart button
        self.restart_button = ctk.CTkButton(
            self.main_frame,
            text="Analyze Another Tweet",
            font=("Arial", 16, "bold"),
            fg_color="#FF3333",
            hover_color="#CC0000",
            corner_radius=30,
            command=self.restart
        )
        self.restart_button.pack(pady=20)
        
    def restart(self):
        # Reset the UI to the initial state
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.setup_initial_ui()
        
    def load_models(self):
        global CNN_MODEL, REDDIT_MODEL, GOLDMAN_MODEL, MARKET_MODEL
        
        # For demonstration, we'll use a simple approach
        # In a real implementation, you would load the actual LoRA models
        print("Loading models...")
        
        # Simulate loading time
        time.sleep(2)
        
        # In a real implementation, you would load the models like this:
        # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        # CNN_MODEL = AutoModelForCausalLM.from_pretrained("path/to/cnn_lora_model")
        # etc.
        
        # For now, we'll just set dummy values
        CNN_MODEL = "loaded"
        REDDIT_MODEL = "loaded"
        GOLDMAN_MODEL = "loaded"
        MARKET_MODEL = "loaded"
        
        print("Models loaded successfully")
        
    def start_analysis(self):
        self.tweet_text = self.tweet_input.get()
        if not self.tweet_text:
            self.tweet_text = "Just bought $10M worth of DogeCoin. The future of currency is here! 🚀"
        
        # Set up loading UI
        self.setup_loading_ui()
        
        # Start analysis in a separate thread
        self.analysis_thread = threading.Thread(target=self.perform_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
    def perform_analysis(self):
        # Simulate processing time for each component
        time.sleep(2)  # CNN News
        
        # Generate CNN headline
        cnn_headline = self.generate_cnn_headline(self.tweet_text)
        
        time.sleep(1.5)  # Reddit Meme
        
        # Generate Reddit meme
        reddit_meme = self.generate_reddit_meme(self.tweet_text, cnn_headline)
        
        # Generate meme image
        self.generate_meme_image(reddit_meme)
        
        time.sleep(1.5)  # Goldman Sachs
        
        # Generate Goldman Sachs report
        goldman_report = self.generate_goldman_report(self.tweet_text)
        
        time.sleep(2)  # Market prediction
        
        # Get stock symbol from tweet
        stock_symbol = self.extract_stock_symbol(self.tweet_text)
        
        # Get current stock price
        current_price = self.get_stock_price(stock_symbol)
        
        # Generate market prediction
        market_prediction = self.generate_market_prediction(
            self.tweet_text, 
            cnn_headline, 
            goldman_report, 
            current_price
        )
        
        # Prepare results
        results = {
            "input": self.tweet_text,
            "title": cnn_headline,
            "stock": stock_symbol,
            "meme": reddit_meme,
            "report": goldman_report,
            "price": current_price,
            "output": market_prediction
        }
        
        # Update UI with results
        self.root.after(0, lambda: self.setup_results_ui(results))
        
    def generate_cnn_headline(self, tweet):
        # In a real implementation, you would use the CNN LoRA model
        # For now, we'll use a simple rule-based approach
        
        if "DogeCoin" in tweet or "dogecoin" in tweet.lower():
            return "Elon Musk's DogeCoin Purchase Sends Crypto Markets Into Frenzy"
        elif "china" in tweet.lower():
            return "TESLA STOPS TAKING NEW ORDERS IN CHINA FOR TWO IMPORTED, US-MADE MODELS"
        elif "tesla" in tweet.lower():
            return "Tesla CEO Announces Major Production Changes Following Tweet"
        elif "spacex" in tweet.lower() or "space" in tweet.lower():
            return "SpaceX Reveals New Mission Details After Musk's Cryptic Tweet"
        else:
            return "Musk's Latest Tweet Sparks Speculation in Tech and Financial Markets"
        
    def generate_reddit_meme(self, tweet, cnn_headline):
        # In a real implementation, you would use the Reddit LoRA model
        # For now, we'll use a simple rule-based approach
        
        if "DogeCoin" in tweet or "dogecoin" in tweet.lower():
            return "Elon holding a literal doge rocket to the moon (50k upvotes)"
        elif "china" in tweet.lower():
            return "When China bans Tesla for no reason: looks like I'll just have to buy China then"
        elif "tesla" in tweet.lower():
            return "Tesla investors watching their stocks after every Elon tweet (30k upvotes)"
        elif "spacex" in tweet.lower() or "space" in tweet.lower():
            return "Elon's plan to colonize Mars is just to escape Twitter (45k upvotes)"
        else:
            return "The market after Elon tweets literally anything (60k upvotes)"
        
    def generate_goldman_report(self, tweet):
        # In a real implementation, you would use the Goldman Sachs LoRA model
        # For now, we'll use a simple rule-based approach
        
        if "DogeCoin" in tweet or "dogecoin" in tweet.lower():
            return "Musk's influence on crypto remains volatile. DOGE-USD may see short-term surge but lacks fundamentals."
        elif "china" in tweet.lower():
            return "Tesla's China strategy shift may impact short-term revenue but strengthen long-term US manufacturing position."
        elif "tesla" in tweet.lower():
            return "Tesla production changes announced via Twitter continue to create market volatility. Institutional investors express governance concerns."
        elif "spacex" in tweet.lower() or "space" in tweet.lower():
            return "SpaceX valuation could see impact from new mission announcements. Private market trading shows increased interest."
        else:
            return "Musk's social media activity continues to create significant market movements. Recommend caution for investors in related securities."
        
    def extract_stock_symbol(self, tweet):
        # Simple logic to extract stock symbol from tweet
        if "DogeCoin" in tweet or "dogecoin" in tweet.lower():
            return "DOGE-USD"
        elif "tesla" in tweet.lower():
            return "TSLA"
        elif "spacex" in tweet.lower():
            return "TSLA"  # Since SpaceX is private, use Tesla as proxy
        else:
            return "TSLA"  # Default to Tesla
        
    def get_stock_price(self, symbol):
        # In a real implementation, you would use an API to get the current stock price
        # For now, we'll use dummy values
        
        try:
            # Try to get real data using yfinance
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except Exception as e:
            print(f"Error fetching stock data: {e}")
        
        # Fallback to dummy values
        if symbol == "DOGE-USD":
            return 0.45
        elif symbol == "TSLA":
            return 241.38
        else:
            return 100.00
        
    def generate_market_prediction(self, tweet, cnn_headline, goldman_report, current_price):
        # In a real implementation, you would use the Market LoRA model
        # For now, we'll use a simple rule-based approach
        
        if "DogeCoin" in tweet or "dogecoin" in tweet.lower():
            # Simulate a spike and then decline
            return [current_price * 1.15, current_price * 1.35, current_price * 1.28, current_price * 1.08, current_price * 0.93]
        elif "china" in tweet.lower():
            # Simulate a decline and then recovery
            return [current_price * 0.97, current_price * 0.93, current_price * 0.95, current_price * 1.02, current_price * 1.05]
        elif "tesla" in tweet.lower():
            # Simulate volatility
            return [current_price * 1.03, current_price * 0.98, current_price * 1.05, current_price * 1.02, current_price * 1.07]
        elif "spacex" in tweet.lower() or "space" in tweet.lower():
            # Simulate steady growth
            return [current_price * 1.01, current_price * 1.03, current_price * 1.05, current_price * 1.08, current_price * 1.10]
        else:
            # Default pattern
            return [current_price * 1.02, current_price * 1.04, current_price * 1.03, current_price * 1.05, current_price * 1.06]
            
    def generate_meme_image(self, meme_text):
        try:
            # In a real implementation, you would use the Pollination.ai API
            # For now, we'll create a simple image with text
            
            # Create a blank image
            img = Image.new('RGB', (500, 500), color=(50, 50, 50))
            d = ImageDraw.Draw(img)
            
            # Add text
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
                
            d.text((20, 20), meme_text, fill=(255, 255, 255), font=font)
            
            # Save the image
            img.save("images/meme.jpg")
            
            print("Meme image generated successfully")
            
        except Exception as e:
            print(f"Error generating meme image: {e}")
            # Create a fallback image
            img = Image.new('RGB', (500, 500), color=(50, 50, 50))
            img.save("images/meme.jpg")

# Main function to run the application
def main():
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = MuskTweetsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
