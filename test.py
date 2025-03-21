import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
from datetime import datetime
import glob

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create directories if they don't exist
os.makedirs('results', exist_ok=True)

# Custom Attention Layer (needed for loading the model)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                               initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                               initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(
            tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Text cleaning function (same as in training script)
def clean_text(text):
    # Handle non-string and missing values
    if not isinstance(text, str):
        return ""
    
    # Define contraction map and emoji map (simplified version from training script)
    contraction_map = {
        "n't": " not", "'s": " is", "'re": " are", "'ve": " have",
        "'d": " would", "'ll": " will", "'m": " am", "ur": " your",
        "won't": "will not", "can't": "cannot"
    }
    
    emoji_map = {
        ':)': ' emoji_smile ', ':(': ' emoji_sad ',
        '<3': ' emoji_heart ', ':D': ' emoji_laugh '
    }
    
    # Contraction replacement
    for cont, exp in contraction_map.items():
        text = text.replace(cont, exp)
    
    # Emoji handling
    for emoji, txt in emoji_map.items():
        text = text.replace(emoji, txt)
    
    # Text cleaning pipeline
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text)  # URLs
    text = re.sub(r'@\w+', ' user ', text)  # Mentions
    text = re.sub(r'#(\w+)', r' hashtag_\1 ', text)  # Hashtags
    text = re.sub(r'[^\w\s]', ' ', text)  # Punctuation
    text = re.sub(r'\b\d+\b', ' number ', text)  # Numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Whitespace
    
    return text

# Function to find the latest model
def find_latest_model():
    print("🔍 Looking for the latest trained model...")
    
    # Get all model files with pattern *_final.keras
    model_files = glob.glob('models/*.keras')
    
    if not model_files:
        print("❌ No trained models found. Please run train.py first.")
        return None, None, None
    
    # Sort by creation time (newest first)
    latest_model_file = max(model_files, key=os.path.getctime)
    model_name = os.path.basename(latest_model_file).replace('.keras', '')
    
    # Find corresponding tokenizer and config files
    tokenizer_file = f'models/{model_name}_tokenizer.pkl'
    config_file = f'models/{model_name}_config.pkl'
    
    if not os.path.exists(tokenizer_file) or not os.path.exists(config_file):
        print(f"❌ Missing tokenizer or config files for model {model_name}")
        return None, None, None
    
    print(f"✅ Found model: {model_name}")
    return latest_model_file, tokenizer_file, config_file

# Function to load model components
def load_model_components():
    # Find latest model
    model_file, tokenizer_file, config_file = find_latest_model()
    
    if not model_file:
        return None, None, None
    
    try:
        # Load model with custom objects
        custom_objects = {'AttentionLayer': AttentionLayer}
        model = load_model(model_file, custom_objects=custom_objects)
        
        # Load tokenizer
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load config
        with open(config_file, 'rb') as f:
            config = pickle.load(f)
        
        print("✅ Model components loaded successfully")
        return model, tokenizer, config
        
    except Exception as e:
        print(f"❌ Error loading model components: {str(e)}")
        return None, None, None

# Function to predict sentiment
def predict_sentiment(text, model, tokenizer, config, save_viz=False):
    print(f"\n🔄 Analyzing text: \"{text}\"")
    
    # Preprocess
    cleaned = clean_text(text)
    print(f"🧹 Cleaned text: \"{cleaned}\"")
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=config['max_len'], padding='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0]
    sentiment_idx = np.argmax(prediction)
    sentiment_label = ['Negative', 'Positive'][sentiment_idx]
    confidence = float(prediction[sentiment_idx])
    
    # Print results
    print(f"📊 Result: {sentiment_label} (Confidence: {confidence:.2%})")
    print(f"   Negative: {prediction[0]:.4f}, Positive: {prediction[1]:.4f}")
    
    # Create visualization
    try:
        # Simple word contribution visualization
        words = cleaned.split()
        if not words:
            print("⚠️ No words found in cleaned text for visualization")
            return {'sentiment': sentiment_label, 'confidence': confidence}
        
        # Assign scores based on sentiment-related words (simplified)
        pos_indicators = ['good', 'great', 'excellent', 'love', 'awesome', 'amazing', 'happy']
        neg_indicators = ['bad', 'terrible', 'awful', 'hate', 'poor', 'horrible', 'worst']
        
        word_scores = {}
        for word in words:
            if word in pos_indicators:
                word_scores[word] = 0.7 if sentiment_idx == 1 else -0.3
            elif word in neg_indicators:
                word_scores[word] = 0.7 if sentiment_idx == 0 else -0.3
            else:
                word_scores[word] = 0.1  # Neutral
        
        # Create visualization
        plt.figure(figsize=(10, max(4, len(words)*0.3)))
        
        # Create color map
        colors = []
        for word in words:
            score = word_scores.get(word, 0)
            if sentiment_idx == 1:  # Positive prediction
                color = cm.Greens(min(0.2 + score, 1.0))
            else:  # Negative prediction
                color = cm.Reds(min(0.2 + score, 1.0))
            colors.append(color)
        
        # Plot with colored bars
        plt.barh(range(len(words)), [word_scores.get(word, 0) for word in words], color=colors)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Contribution to Sentiment')
        plt.title(f'Predicted: {sentiment_label} (Confidence: {confidence:.2%})')
        plt.tight_layout()
        
        # Save visualization if requested
        if save_viz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/sentiment_analysis_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {filename}")
        
        # Show visualization
        plt.show()
            
    except Exception as e:
        print(f"⚠️ Error generating visualization: {str(e)}")
    
    return {'sentiment': sentiment_label, 'confidence': confidence}

# Main function for interactive testing
def main():
    print("=" * 60)
    print("📝 SENTIMENT ANALYSIS TESTING TOOL")
    print("=" * 60)
    
    # Load model components
    model, tokenizer, config = load_model_components()
    
    if not model:
        return
    
    print("\n🔍 Model Information:")
    print(f"   Model type: {config['model_type']}")
    print(f"   Vocabulary size: {config['max_words']}")
    print(f"   Max sequence length: {config['max_len']}")
    
    # Interactive loop
    print("\n💬 Enter text to analyze sentiment (or 'q' to quit):")
    
    while True:
        text = input("\n> ")
        
        if text.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        if not text.strip():
            print("⚠️ Please enter some text to analyze")
            continue
        
        # Analyze sentiment
        result = predict_sentiment(text, model, tokenizer, config, save_viz=False)
        
        # Ask if user wants to save the visualization
        save_viz = input("\nSave visualization? (y/n): ").lower() == 'y'
        if save_viz:
            predict_sentiment(text, model, tokenizer, config, save_viz=True)
        
        print("-" * 60)

# Alternative batch testing function
def batch_test(texts):
    # Load model components
    model, tokenizer, config = load_model_components()
    
    if not model:
        return
    
    results = []
    
    for text in texts:
        result = predict_sentiment(text, model, tokenizer, config, save_viz=False)
        results.append({
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence']
        })
    
    return results

if __name__ == "__main__":
    main()