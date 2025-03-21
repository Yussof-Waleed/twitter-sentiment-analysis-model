import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_curve, f1_score,
                             roc_curve, auc, precision_score, recall_score)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout, 
                                   Bidirectional, BatchNormalization,
                                   SpatialDropout1D, GlobalMaxPooling1D, 
                                   Conv1D, Attention, Input, Concatenate,
                                   GRU, Layer, MultiHeadAttention)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.optimizers import Adam
import pickle
import time
from datetime import datetime
import matplotlib.cm as cm
import tqdm
import io
from sklearn.utils import resample
from collections import Counter


# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Create directories for outputs
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Enhanced configuration parameters
CONFIG = {
    'max_words': 30000,            # Increased vocabulary size
    'max_len': 150,                # Increased sequence length
    'embedding_dim': 300,          # Increased embedding dimension
    'batch_size': 64,
    'epochs': 20,                  # Increased epochs with early stopping
    'use_class_weights': True,
    'enable_augmentation': True,   # Enable data augmentation
    'k_folds': 5,                  # K-fold cross-validation
    'model_type': 'attention_gru', # Options: 'bilstm', 'attention_gru', 'cnn_rnn'
    'learning_rate': 0.001,
    'dropout_rate': 0.4,
    'l1_reg': 0.0001,
    'l2_reg': 0.001,
    'random_seed': 42,
    'validation_split': 0.15,
    'test_split': 0.15,
    'use_glove': False,            # Set to True if GloVe embeddings are available
    'glove_path': './glove.twitter.27B.100d.txt',
}

print("🔍 Checking available GPUs...")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        print(f"✅ Found {len(gpus)} GPU(s): {gpus}")
        
        # Configure TensorFlow to use memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🔧 Enabled memory growth for {gpu}")
        
        # Set GPU as preferred device but don't hide CPU
        tf.config.set_visible_devices(gpus, 'GPU')
        
        # Create mixed precision policy for faster training
        mixed_precision = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(mixed_precision)
        print("🚀 Enabled mixed precision training with float16")
        
        # Verify GPU is being used
        print(f"💻 Is GPU available: {tf.test.is_gpu_available()}")
        print(f"💻 Is built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"💻 Using device: {tf.test.gpu_device_name()}")
    except RuntimeError as e:
        print(f"❌ GPU error: {e}")
else:
    print("❌ No GPUs found. Running on CPU only.")


# Set random seeds for reproducibility
np.random.seed(CONFIG['random_seed'])
tf.random.set_seed(CONFIG['random_seed'])

# Custom Attention Layer
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

# Step 1: Data Download and Preparation with better logging
def download_twitter_data(file_path='twitter_data.csv'):
    print("🔄 Loading Twitter data...")
    
    try:
        # Read CSV with proper handling
        df = pd.read_csv(
            './xdata.csv',
            encoding='latin-1',
            names=['sentiment', 'tweet_id', 'date', 'query', 'username', 'text'],
            header=None,
            # nrows=40000  # You can adjust this or add a parameter to control it
        )
        
        # Clean initial data
        df = df[['text', 'sentiment']].dropna()
        df['sentiment'] = df['sentiment'].astype(int)
        
        # Shuffle the dataset to ensure random sampling
        df = df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)

        # Set maximum size for the balanced dataset
        max_dataset_size = 100000

        # Determine number of samples per sentiment class
        num_classes = len(df['sentiment'].unique())
        samples_per_class = max_dataset_size // num_classes

        # Create a balanced dataset with equal representation from each class
        print(f"🔄 Balancing dataset: targeting {samples_per_class} samples per class...")
        balanced_df = pd.DataFrame()

        for sentiment in df['sentiment'].unique():
            # Extract samples for current sentiment
            sentiment_df = df[df['sentiment'] == sentiment]
            available_samples = len(sentiment_df)
            
            # Sample or take all available data
            if available_samples > samples_per_class:
                sentiment_df = sentiment_df.sample(samples_per_class, random_state=CONFIG['random_seed'])
                print(f"  Class {sentiment}: sampled {samples_per_class} from {available_samples} available")
            else:
                print(f"  Class {sentiment}: using all {available_samples} available samples (less than target)")
            
            # Add to balanced dataset
            balanced_df = pd.concat([balanced_df, sentiment_df])

        # Shuffle the final balanced dataset
        df = balanced_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
        
        # Print the final size of the dataset
        print(f"✅ Final balanced dataset size: {len(df)} samples ({samples_per_class} per class)")
        # Get equal number of samples from each sentiment class
        sentiment_counts = df['sentiment'].value_counts()
        min_count = sentiment_counts.min()
        
        # Create a balanced dataset with equal representation from each class
        balanced_df = pd.DataFrame()
        for sentiment in df['sentiment'].unique():
            sentiment_df = df[df['sentiment'] == sentiment].sample(min_count, random_state=CONFIG['random_seed'])
            balanced_df = pd.concat([balanced_df, sentiment_df])
        
        # Shuffle the balanced dataset again
        df = balanced_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)


        
        # Log dataset statistics
        print(f"✅ Dataset loaded successfully: {len(df)} entries")
        print(f"🔍 Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        # Save processed file
        df.to_csv(file_path, index=False)
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

# Step 2: Enhanced Text Preprocessing
# Extended contraction map
contraction_map = {
    "n't": " not", "'s": " is", "'re": " are", "'ve": " have",
    "'d": " would", "'ll": " will", "'m": " am", "ur": " your",
    "won't": "will not", "can't": "cannot", "'cause": "because",
    "ain't": "am not", "gonna": "going to", "wanna": "want to",
    "gotta": "got to", "y'all": "you all", "kinda": "kind of",
    "sorta": "sort of", "lemme": "let me", "gimme": "give me"
}

# Extended emoji map
emoji_map = {
    ':)': ' emoji_smile ', ':(': ' emoji_sad ',
    '<3': ' emoji_heart ', ':/': ' emoji_confused ',
    ':D': ' emoji_laugh ', ":'(": ' emoji_cry ',
    ':p': ' emoji_tongue ', ':P': ' emoji_tongue ',
    ':o': ' emoji_surprised ', ':O': ' emoji_surprised ',
    ';)': ' emoji_wink ', ':*': ' emoji_kiss ',
    '>:(': ' emoji_angry ', '-_-': ' emoji_expressionless ',
    '=)': ' emoji_smile ', ':|': ' emoji_neutral ',
    '._.' : ' emoji_awkward '
}

def clean_text(text):
    # Handle non-string and missing values
    if not isinstance(text, str):
        return ""
    
    # Contraction replacement
    for cont, exp in contraction_map.items():
        text = text.replace(cont, exp)
    
    # Emoji handling
    for emoji, txt in emoji_map.items():
        text = text.replace(emoji, txt)
    
    # Text cleaning pipeline
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text)  # URLs replaced with token
    text = re.sub(r'@\w+', ' user ', text)  # Mentions replaced with token
    text = re.sub(r'#(\w+)', r' hashtag_\1 ', text)  # Hashtags preserved but marked
    text = re.sub(r'[^\w\s]', ' ', text)  # Punctuation
    text = re.sub(r'\b\d+\b', ' number ', text)  # Numbers replaced with token
    text = re.sub(r'\s+', ' ', text).strip()  # Whitespace
    
    return text

def preprocess_data(df):
    print("🔄 Preprocessing data...")
    start_time = time.time()
    
    df['text'] = df['text'].fillna("").astype(str)
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Tokenization and normalization
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'rt', 'amp', 'via'}  # Common Twitter noise
    stop_words.update(custom_stopwords)
    
    lemmatizer = WordNetLemmatizer()
    
    # Progress tracking for long operations
    print("Tokenizing and lemmatizing texts...")
    
    def process_text(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join([
            lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in stop_words and len(word) > 2
        ])
    
    # Apply processing with progress tracking
    df['processed_text'] = [process_text(text) for text in tqdm.tqdm(df['cleaned_text'])]
    
    # Filter out empty processed texts
    empty_count = df['processed_text'].str.strip().eq('').sum()
    if empty_count > 0:
        print(f"⚠️ Removing {empty_count} entries with empty processed text")
        df = df[df['processed_text'].str.strip() != '']
    
    processing_time = time.time() - start_time
    print(f"✅ Preprocessing completed in {processing_time:.2f} seconds")
    
    # Add text length as a feature
    df['text_length'] = df['processed_text'].apply(lambda x: len(x.split()))
    
    return df

# Step 3: Enhanced Data Augmentation
def augment_dataset(df):
    if not CONFIG['enable_augmentation']:
        return df
    
    print("🔄 Augmenting data...")
    
    # Calculate class distribution
    class_counts = df['sentiment'].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    print(f"Class distribution before augmentation: {class_counts.to_dict()}")
    
    # Get minority and majority samples
    minority_df = df[df['sentiment'] == minority_class]
    majority_df = df[df['sentiment'] == majority_class]
    
    # 1. Synonym replacement
    print("Applying synonym replacement...")
    
    def synonym_replacement(text):
        words = text.split()
        if len(words) <= 3:
            return text
            
        # Replace 10-30% of words with synonyms
        n_to_replace = max(1, int(len(words) * np.random.uniform(0.1, 0.3)))
        replace_indices = np.random.choice(range(len(words)), n_to_replace, replace=False)
        
        for idx in replace_indices:
            word = words[idx]
            synonyms = []
            
            # Simple synonym generation (in production, use WordNet or similar)
            if word == 'good': synonyms = ['great', 'excellent', 'nice', 'fine']
            elif word == 'bad': synonyms = ['poor', 'terrible', 'awful', 'horrible']
            elif word == 'happy': synonyms = ['glad', 'pleased', 'content', 'joyful']
            elif word == 'sad': synonyms = ['unhappy', 'upset', 'down', 'miserable']
            
            if synonyms:
                words[idx] = np.random.choice(synonyms)
        
        return ' '.join(words)
    
    # 2. Random word deletion
    def random_deletion(text):
        words = text.split()
        if len(words) <= 5:
            return text
            
        # Delete 10-20% of words
        keep_prob = np.random.uniform(0.8, 0.9)
        new_words = [word for word in words if np.random.random() < keep_prob]
        
        return ' '.join(new_words) if new_words else text
    
    # 3. Random swap
    def random_swap(text):
        words = text.split()
        if len(words) <= 3:
            return text
            
        # Swap 1-3 pairs of words
        n_swaps = min(3, len(words) // 2)
        for _ in range(np.random.randint(1, n_swaps + 1)):
            idx1, idx2 = np.random.choice(range(len(words)), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    # Apply augmentation to minority class
    augmented_texts = []
    augmented_sentiments = []
    
    # Determine how many samples to generate
    n_to_generate = min(len(majority_df) - len(minority_df), len(minority_df) * 2)
    
    # Generate augmented samples
    for _ in range(n_to_generate):
        # Randomly select a minority sample
        sample = minority_df.sample(1).iloc[0]
        text = sample['processed_text']
        sentiment = sample['sentiment']
        
        # Randomly choose augmentation technique
        aug_type = np.random.choice(['synonym', 'deletion', 'swap'])
        
        if aug_type == 'synonym':
            new_text = synonym_replacement(text)
        elif aug_type == 'deletion':
            new_text = random_deletion(text)
        else:
            new_text = random_swap(text)
        
        augmented_texts.append(new_text)
        augmented_sentiments.append(sentiment)
    
    # Create augmented dataframe
    aug_df = pd.DataFrame({
        'text': ['[AUGMENTED]'] * len(augmented_texts),
        'sentiment': augmented_sentiments,
        'cleaned_text': augmented_texts,
        'processed_text': augmented_texts,
        'text_length': [len(text.split()) for text in augmented_texts]
    })
    
    # Combine with original data
    combined_df = pd.concat([df, aug_df]).reset_index(drop=True)
    
    print(f"Class distribution after augmentation: {combined_df['sentiment'].value_counts().to_dict()}")
    print(f"✅ Added {len(aug_df)} augmented samples")
    
    return combined_df

# Step 4: Enhanced Data Analysis and Visualization
def analyze_data(df):
    print("🔄 Analyzing data...")
    
    # Create figure for all visualizations
    plt.figure(figsize=(20, 16))
    
    # 1. Sentiment distribution
    plt.subplot(2, 2, 1)
    sentiment_counts = df['sentiment'].value_counts().sort_index()
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    for i, count in enumerate(sentiment_counts):
        ax.text(i, count + 50, f"{count}", ha='center')
    plt.title('Class Distribution', fontsize=14)
    plt.xlabel('Sentiment Class')
    plt.ylabel('Count')
    
    # 2. Text length analysis by sentiment
    plt.subplot(2, 2, 2)
    sns.boxplot(x='sentiment', y='text_length', data=df)
    plt.title('Text Length by Sentiment', fontsize=14)
    plt.xlabel('Sentiment Class')
    plt.ylabel('Word Count')
    
    # 3. Text length distribution
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='text_length', hue='sentiment', bins=50, kde=True, element="step")
    plt.title('Text Length Distribution', fontsize=14)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    # 4. Word count distribution
    plt.subplot(2, 2, 4)
    
    # Get word frequencies
    all_words = ' '.join(df['processed_text']).split()
    word_counts = Counter(all_words)
    top_words = dict(word_counts.most_common(20))
    
    # Plot word frequencies
    sns.barplot(x=list(top_words.keys()), y=list(top_words.values()))
    plt.title('Top 20 Words', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save comprehensive analysis
    plt.savefig('images/data_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate word clouds for each sentiment
    try:
        from wordcloud import WordCloud
        
        plt.figure(figsize=(16, 8))
        
        for i, sentiment in enumerate(df['sentiment'].unique()):
            plt.subplot(1, len(df['sentiment'].unique()), i+1)
            text = ' '.join(df[df['sentiment'] == sentiment]['processed_text'])
            wordcloud = WordCloud(
                width=400, height=400, 
                background_color='white',
                colormap='Blues',
                max_words=100,
                contour_width=1
            ).generate(text)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f"Sentiment {sentiment} Word Cloud")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('images/sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("WordCloud package not installed. Skipping word cloud visualization.")
    
    print("✅ Data analysis visualizations saved to 'visualizations' folder")
    
    return df

# Step 5: Enhanced Data Preparation
def prepare_data(df):
    print("🔄 Preparing data for training...")
    
    # Tokenization and sequencing
    tokenizer = Tokenizer(num_words=CONFIG['max_words'], oov_token='<OOV>')
    tokenizer.fit_on_texts(df['processed_text'])
    
    # Save the fitted tokenizer
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ Tokenizer saved to 'models/tokenizer.pickle'")
    
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    padded_sequences = pad_sequences(sequences, maxlen=CONFIG['max_len'], padding='post')

    # Print vocabulary stats
    word_index = tokenizer.word_index
    print(f"Total unique words: {len(word_index)}")
    print(f"Using top {min(CONFIG['max_words'], len(word_index))} words")
    
    # Convert labels to one-hot encoding
    labels = pd.get_dummies(df['sentiment']).values
    
    # Create stratified train-validation-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        padded_sequences, labels, 
        test_size=CONFIG['test_split'], 
        stratify=df['sentiment'],
        random_state=CONFIG['random_seed']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=CONFIG['validation_split']/(1-CONFIG['test_split']),
        stratify=pd.Series(np.argmax(y_temp, axis=1)),
        random_state=CONFIG['random_seed']
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Prepare embedding matrix if using pre-trained embeddings
    embedding_matrix = None
    if CONFIG['use_glove']:
        print("Loading GloVe embeddings...")
        try:
            embeddings_index = {}
            with open(CONFIG['glove_path'], encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            
            print(f"Loaded {len(embeddings_index)} word vectors")
            
            # Prepare embedding matrix
            embedding_matrix = np.zeros((min(CONFIG['max_words'], len(word_index) + 1), 
                                       CONFIG['embedding_dim']))
            
            for word, i in word_index.items():
                if i >= CONFIG['max_words']:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    
            print(f"Embedding matrix shape: {embedding_matrix.shape}")
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            CONFIG['use_glove'] = False
    
    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, embedding_matrix

# Step 6: Enhanced Model Architecture
def build_model(tokenizer, embedding_matrix=None):
    print(f"🔄 Building {CONFIG['model_type']} model...")
    
    vocab_size = min(CONFIG['max_words'], len(tokenizer.word_index) + 1)
    
    regularizer = l1_l2(l1=CONFIG['l1_reg'], l2=CONFIG['l2_reg'])
    
    # Initialize embedding_layer
    if CONFIG['use_glove'] and embedding_matrix is not None:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=CONFIG['embedding_dim'],
            weights=[embedding_matrix],
            input_length=CONFIG['max_len'],
            trainable=False
        )
    else:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=CONFIG['embedding_dim'],
            input_length=CONFIG['max_len']
        )
    
    # Choose model architecture based on configuration
    if CONFIG['model_type'] == 'bilstm':
        # Bidirectional LSTM model
        model = Sequential([
            embedding_layer,
            SpatialDropout1D(CONFIG['dropout_rate']),
            
            Bidirectional(LSTM(
                128, 
                return_sequences=True,
                kernel_regularizer=regularizer
            )),
            BatchNormalization(),
            Dropout(CONFIG['dropout_rate']),
            
            Bidirectional(LSTM(
                64, 
                kernel_regularizer=regularizer
            )),
            BatchNormalization(),
            Dropout(CONFIG['dropout_rate']),
            
            Dense(64, activation='relu', kernel_regularizer=regularizer),
            Dropout(CONFIG['dropout_rate']),
            
            Dense(2, activation='softmax')
        ])
        
    elif CONFIG['model_type'] == 'attention_gru':
        # GRU with Attention model (functional API)
        inputs = Input(shape=(CONFIG['max_len'],))
        x = embedding_layer(inputs)
        x = SpatialDropout1D(CONFIG['dropout_rate'])(x)
        
        # First bidirectional GRU layer
        gru1 = Bidirectional(GRU(128, return_sequences=True, 
                                kernel_regularizer=regularizer))(x)
        gru1 = BatchNormalization()(gru1)
        
        # Second bidirectional GRU layer
        gru2 = Bidirectional(GRU(64, return_sequences=True,
                                kernel_regularizer=regularizer))(gru1)
        gru2 = BatchNormalization()(gru2)
        
        # Custom attention layer
        attention = AttentionLayer()(gru2)
        
        # Dense layers
        dense1 = Dense(64, activation='relu', kernel_regularizer=regularizer)(attention)
        dropout = Dropout(CONFIG['dropout_rate'])(dense1)
        outputs = Dense(2, activation='softmax')(dropout)
        
        model = Model(inputs=inputs, outputs=outputs)
        
    elif CONFIG['model_type'] == 'cnn_rnn':
        # CNN-RNN hybrid model
        inputs = Input(shape=(CONFIG['max_len'],))
        x = embedding_layer(inputs)
        x = SpatialDropout1D(CONFIG['dropout_rate'])(x)
        
        # CNN layers with different kernel sizes
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        conv2 = Conv1D(filters=64, kernel_size=4, padding='same', activation='relu')(x)
        conv3 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x)
        
        # Concatenate CNN outputs
        concat = Concatenate()([conv1, conv2, conv3])
        
        # GRU layer
        gru = Bidirectional(GRU(128, return_sequences=True))(concat)
        
        # Global pooling
        pooled = GlobalMaxPooling1D()(gru)
        
        # Dense layers
        dense = Dense(64, activation='relu', kernel_regularizer=regularizer)(pooled)
        dropout = Dropout(CONFIG['dropout_rate'])(dense)
        outputs = Dense(2, activation='softmax')(dropout)
        
        model = Model(inputs=inputs, outputs=outputs)
    
    else:
        raise ValueError(f"Unknown model type: {CONFIG['model_type']}")
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=CONFIG['learning_rate'],
        decay_steps=10000,
        decay_rate=0.95)
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    # Model summary
    model.summary()
    
    # Visual representation of model architecture
    try:
        tf.keras.utils.plot_model(
            model, 
            to_file='images/model_architecture.png',
            show_shapes=True, 
            show_dtype=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True
        )
        print("✅ Model architecture visualization saved")
    except Exception as e:
        print(f"⚠️ Unable to generate model visualization: {str(e)}")
    
    # Create RNN visualization
    create_rnn_visualization()
    
    return model

# Create a visualization of RNN structure
def create_rnn_visualization():
    plt.figure(figsize=(12, 8))
    
    # Drawing the RNN cell
    def draw_rnn_cell(ax, x, y, width=0.8, height=0.6, title="GRU/LSTM Cell"):
        # Main cell rectangle
        rect = mpatches.Rectangle((x, y), width, height, 
                                 fill=True, facecolor='lightblue', 
                                 edgecolor='blue', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Draw internal gates
        gate_width = width/4
        gate_height = height/3
        gate_y = y + height/3
        
        # Input gate
        input_gate = mpatches.Rectangle((x + width*0.1, gate_y), gate_width, gate_height, 
                                      fill=True, facecolor='yellow', alpha=0.7)
        ax.add_patch(input_gate)
        ax.text(x + width*0.1 + gate_width/2, gate_y + gate_height/2, "i", 
               ha='center', va='center', fontsize=10)
        
        # Forget gate
        forget_gate = mpatches.Rectangle((x + width*0.3 + gate_width, gate_y), 
                                       gate_width, gate_height, 
                                       fill=True, facecolor='orange', alpha=0.7)
        ax.add_patch(forget_gate)
        ax.text(x + width*0.3 + gate_width*1.5, gate_y + gate_height/2, "f", 
               ha='center', va='center', fontsize=10)
        
        # Output gate
        output_gate = mpatches.Rectangle((x + width*0.5 + gate_width*2, gate_y), 
                                       gate_width, gate_height, 
                                       fill=True, facecolor='lightgreen', alpha=0.7)
        ax.add_patch(output_gate)
        ax.text(x + width*0.5 + gate_width*2.5, gate_y + gate_height/2, "o", 
               ha='center', va='center', fontsize=10)
        
        # Title
        ax.text(x + width/2, y + height + 0.05, title, 
               ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Create a new axis
    ax = plt.subplot(111)
    
    # Set limits and turn off axis
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Draw title
    plt.title(f"{CONFIG['model_type'].upper()} Architecture for Sentiment Analysis", 
             fontsize=16, fontweight='bold')
    
    # Draw input
    ax.text(0.5, 2.7, "Input Text", fontsize=14, ha='center')
    input_rect = mpatches.Rectangle((0.2, 2.3), 0.6, 0.3, 
                                  fill=True, facecolor='lightgray', alpha=0.7)
    ax.add_patch(input_rect)
    
    # Draw embedding layer
    ax.text(1.3, 2.7, "Embedding Layer", fontsize=14, ha='center')
    embed_rect = mpatches.Rectangle((1.0, 2.3), 0.6, 0.3, 
                                  fill=True, facecolor='lightyellow', alpha=0.7)
    ax.add_patch(embed_rect)
    
    # Draw arrow from input to embedding
    ax.arrow(0.8, 2.45, 0.15, 0, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
    
    # Draw RNN cells
    if CONFIG['model_type'] == 'bilstm':
        # Draw forward LSTM
        draw_rnn_cell(ax, 0.6, 1.4, title="LSTM Forward")
        # Draw backward LSTM
        draw_rnn_cell(ax, 2.0, 1.4, title="LSTM Backward")
        # Title for bidirectional
        ax.text(1.7, 1.9, "Bidirectional LSTM", fontsize=14, ha='center')
    elif CONFIG['model_type'] == 'attention_gru':

        draw_rnn_cell(ax, 0.6, 1.4, title="GRU Forward")
        # Draw backward GRU
        draw_rnn_cell(ax, 2.0, 1.4, title="GRU Backward")
        # Title for bidirectional
        ax.text(1.7, 1.9, "Bidirectional GRU", fontsize=14, ha='center')
        # Draw attention mechanism
        attention_rect = mpatches.Rectangle((1.2, 0.8), 1.2, 0.3, 
                                          fill=True, facecolor='lightcoral', alpha=0.7)
        ax.add_patch(attention_rect)
        ax.text(1.8, 0.95, "Attention Layer", fontsize=14, ha='center')
    elif CONFIG['model_type'] == 'cnn_rnn':
        # Draw CNN layers
        cnn_rect1 = mpatches.Rectangle((0.3, 1.6), 0.7, 0.3, 
                                     fill=True, facecolor='lightgreen', alpha=0.7)
        ax.add_patch(cnn_rect1)
        ax.text(0.65, 1.75, "CNN (kernel=3)", fontsize=10, ha='center')
        
        cnn_rect2 = mpatches.Rectangle((1.2, 1.6), 0.7, 0.3, 
                                     fill=True, facecolor='lightgreen', alpha=0.7)
        ax.add_patch(cnn_rect2)
        ax.text(1.55, 1.75, "CNN (kernel=4)", fontsize=10, ha='center')
        
        cnn_rect3 = mpatches.Rectangle((2.1, 1.6), 0.7, 0.3, 
                                     fill=True, facecolor='lightgreen', alpha=0.7)
        ax.add_patch(cnn_rect3)
        ax.text(2.45, 1.75, "CNN (kernel=5)", fontsize=10, ha='center')
        
        # Draw concatenate layer
        concat_rect = mpatches.Rectangle((0.8, 1.2), 1.5, 0.2, 
                                       fill=True, facecolor='lightsalmon', alpha=0.7)
        ax.add_patch(concat_rect)
        ax.text(1.55, 1.3, "Concatenate", fontsize=12, ha='center')
        
        # Draw GRU
        draw_rnn_cell(ax, 1.15, 0.6, title="Bidirectional GRU")
    
    # Draw output layers
    dense_rect = mpatches.Rectangle((2.8, 1.0), 0.7, 0.3, 
                                  fill=True, facecolor='lightblue', alpha=0.7)
    ax.add_patch(dense_rect)
    ax.text(3.15, 1.15, "Dense Layers", fontsize=12, ha='center')
    
    output_rect = mpatches.Rectangle((3.6, 0.5), 0.7, 0.3, 
                                   fill=True, facecolor='lightpink', alpha=0.7)
    ax.add_patch(output_rect)
    ax.text(3.95, 0.65, "Softmax Output", fontsize=12, ha='center')
    
    # Draw arrows
    ax.arrow(1.3, 2.3, 0, -0.2, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
    if CONFIG['model_type'] == 'cnn_rnn':
        # Arrows from embedding to CNN
        ax.arrow(1.3, 2.1, -0.3, -0.15, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        ax.arrow(1.3, 2.1, 0.25, -0.15, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        ax.arrow(1.3, 2.1, 0.8, -0.15, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        
        # Arrows from CNN to concatenate
        ax.arrow(0.65, 1.6, 0.3, -0.25, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        ax.arrow(1.55, 1.6, 0, -0.2, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        ax.arrow(2.45, 1.6, -0.3, -0.25, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        
        # Arrow from concatenate to GRU
        ax.arrow(1.55, 1.2, 0, -0.25, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        
        # Arrow from GRU to dense
        ax.arrow(1.95, 0.8, 0.8, 0.1, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
    else:
        # Arrows from embedding to RNN
        ax.arrow(1.3, 2.1, -0.3, -0.3, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        ax.arrow(1.3, 2.1, 0.8, -0.3, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        
        # Arrows from RNN to attention/dense
        if CONFIG['model_type'] == 'attention_gru':
            ax.arrow(1.0, 1.4, 0.5, -0.3, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
            ax.arrow(2.4, 1.4, -0.5, -0.3, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
            ax.arrow(1.8, 0.8, 0.95, 0.1, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        else:
            ax.arrow(1.0, 1.4, 1.75, -0.2, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
            ax.arrow(2.4, 1.4, 0.35, -0.2, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
    
    # Arrow from dense to output
    ax.arrow(3.15, 1.0, 0.4, -0.2, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
    
    # Add legend
    legend_items = []
    legend_labels = []
    
    # Embedding
    embed_patch = mpatches.Patch(color='lightyellow', alpha=0.7, label='Embedding')
    legend_items.append(embed_patch)
    legend_labels.append('Embedding')
    
    # RNN
    rnn_patch = mpatches.Patch(color='lightblue', alpha=0.7, label='RNN/Dense')
    legend_items.append(rnn_patch)
    legend_labels.append('RNN/Dense')
    
    if CONFIG['model_type'] == 'attention_gru':
        # Attention
        att_patch = mpatches.Patch(color='lightcoral', alpha=0.7, label='Attention')
        legend_items.append(att_patch)
        legend_labels.append('Attention')
    
    if CONFIG['model_type'] == 'cnn_rnn':
        # CNN
        cnn_patch = mpatches.Patch(color='lightgreen', alpha=0.7, label='CNN')
        legend_items.append(cnn_patch)
        legend_labels.append('CNN')
        
        # Concatenate
        concat_patch = mpatches.Patch(color='lightsalmon', alpha=0.7, label='Concatenate')
        legend_items.append(concat_patch)
        legend_labels.append('Concatenate')
    
    # Output
    output_patch = mpatches.Patch(color='lightpink', alpha=0.7, label='Output')
    legend_items.append(output_patch)
    legend_labels.append('Output')
    
    plt.legend(legend_items, legend_labels, loc='upper right')
    
    # Add explanatory text
    if CONFIG['model_type'] == 'bilstm':
        explainer = (
            "Bidirectional LSTM Architecture:\n"
            "1. Text input is embedded into vector space\n"
            "2. Bidirectional LSTM processes sequences in both directions\n"
            "3. Dense layers interpret the features\n"
            "4. Softmax provides sentiment probabilities"
        )
    elif CONFIG['model_type'] == 'attention_gru':
        explainer = (
            "Attention GRU Architecture:\n"
            "1. Text input is embedded into vector space\n"
            "2. Bidirectional GRU processes sequences in both directions\n"
            "3. Attention layer focuses on important parts of sequence\n"
            "4. Dense layers interpret features with attention weights\n"
            "5. Softmax provides sentiment probabilities"
        )
    else:
        explainer = (
            "CNN-RNN Hybrid Architecture:\n"
            "1. Text input is embedded into vector space\n"
            "2. Multiple parallel CNNs extract n-gram features\n"
            "3. Features are concatenated\n"
            "4. Bidirectional GRU processes sequential patterns\n"
            "5. Dense layers interpret combined features\n"
            "6. Softmax provides sentiment probabilities"
        )
    
    ax.text(0.5, 0.3, explainer, fontsize=10, va='center', ha='left')
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig('images/rnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ RNN architecture visualization saved")

# Step 7: Enhanced Model Training
def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    print("🔄 Training model on GPU...")
    
    # Print device placement information
    print(f"Training data on device: {tf.debugging.set_log_device_placement(True)}")
    with tf.device('/GPU:0'):
        
        # Create a unique model name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{CONFIG['model_type']}_{timestamp}"
        
        # Create callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            
            # Model checkpoint to save after each epoch
            ModelCheckpoint(
                filepath=f"models/{model_name}_epoch_{{epoch:02d}}.keras",
                monitor='val_loss',
                save_best_only=False,
                verbose=1,
                mode='min',
                save_freq='epoch'
            ),

            # Learning rate reducer
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-6,
                mode='min'
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=f"logs/{model_name}",
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            callbacks=callbacks,
            class_weight=class_weights if CONFIG['use_class_weights'] else None,
            verbose=1
        )
        
        # Save final model
        model.save(f"models/{model_name}_final.keras")
        print(f"✅ Model saved as 'models/{model_name}_final.keras'")
        
        # Visualize training history
        visualize_training_history(history, model_name)
    
    return history, model_name

# Visualize training history
def visualize_training_history(history, model_name):
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'], label='Train')
    plt.plot(history.history['val_precision'], label='Validation')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'], label='Train')
    plt.plot(history.history['val_recall'], label='Validation')
    plt.title('Model Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'images/{model_name}_training_history.png', dpi=300)
    plt.close()
    print(f"✅ Training history visualization saved")

# Step 8: Enhanced Model Evaluation
def evaluate_model(model, X_test, y_test, tokenizer, model_name):
    print("🔄 Evaluating model...")
    
    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    
    # Print metrics
    print("\n📊 Performance Metrics:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    class_report = classification_report(y_true, y_pred_classes)
    print(class_report)
    
    # Save metrics to file
    with open(f'models/{model_name}_metrics.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
    
    # Create visualizations
    plt.figure(figsize=(20, 10))
    
    # 1. Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # 2. Precision-Recall curve
    plt.subplot(2, 2, 2)
    for i in range(y_test.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_prob[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {i}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # 3. ROC curve
    plt.subplot(2, 2, 3)
    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # 4. Confidence Distribution
    plt.subplot(2, 2, 4)
    for i in range(y_test.shape[1]):
        class_pred_prob = y_pred_prob[y_true == i, i]
        if len(class_pred_prob) > 0:
            sns.histplot(class_pred_prob, bins=20, alpha=0.5, label=f'Class {i}')
    
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
    plt.xlim([0, 1])
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution by True Class')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'images/{model_name}_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Evaluation visualizations saved")
    
    # Error analysis
    error_analysis(X_test, y_true, y_pred_classes, y_pred_prob, tokenizer, model_name)
    
    return accuracy, precision, recall, f1

# Error Analysis
def error_analysis(X_test, y_true, y_pred, y_pred_prob, tokenizer, model_name):
    print("🔄 Performing error analysis...")
    
    # Get indices of misclassified samples
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples found!")
        return
    
    # Get a sample of misclassified examples (up to 20)
    sample_size = min(20, len(misclassified_indices))
    sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)
    
    # Convert sequences back to text
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    # Prepare data for misclassification analysis
    misclassified_data = []
    for idx in sample_indices:
        # Get the sequence
        sequence = X_test[idx]
        # Convert sequence to words
        words = [reverse_word_index.get(i, '') for i in sequence if i != 0]
        text = ' '.join(words)
        
        # Get true and predicted classes and confidence
        true_class = y_true[idx]
        pred_class = y_pred[idx]
        confidence = y_pred_prob[idx, pred_class]
        
        misclassified_data.append({
            'text': text,
            'true_class': true_class,
            'pred_class': pred_class,
            'confidence': confidence
        })
    
    # Save error analysis to file
    with open(f'models/{model_name}_error_analysis.txt', 'w') as f:
        f.write(f"Error Analysis - {len(misclassified_indices)} misclassified samples out of {len(y_true)}\n")
        f.write(f"Error rate: {len(misclassified_indices) / len(y_true):.2%}\n\n")
        
        f.write("Sample of misclassified texts:\n")
        f.write("-" * 80 + "\n")
        
        for i, data in enumerate(misclassified_data):
            f.write(f"Example {i+1}:\n")
            f.write(f"Text: {data['text'][:100]}...\n")
            f.write(f"True Class: {data['true_class']}, Predicted: {data['pred_class']}, Confidence: {data['confidence']:.2%}\n")
            f.write("-" * 80 + "\n")
    
    print(f"✅ Error analysis saved to 'models/{model_name}_error_analysis.txt'")
    
    # Create confusion visualization
    plt.figure(figsize=(10, 8))
    
    # Group by confidence
    confidence_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = ['<60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    # Count mistakes by confidence
    confidences = y_pred_prob[misclassified_indices, y_pred[misclassified_indices]]
    # Use full bin array and don't subtract 1
    binned = np.digitize(confidences, confidence_bins) - 1
    # Clip to avoid negative values
    binned = np.clip(binned, 0, len(bin_labels)-1)
    mistake_counts = np.bincount(binned, minlength=len(bin_labels))
    
    # Plot
    plt.bar(bin_labels, mistake_counts, color='crimson', alpha=0.7)
    plt.title('Misclassifications by Confidence Level')
    plt.xlabel('Confidence Level')
    plt.ylabel('Count of Errors')
    plt.grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(mistake_counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(f'images/{model_name}_error_by_confidence.png', dpi=300)
    plt.close()
    print("✅ Error confidence visualization saved")

# Step 9: Save Components with versioning
def save_components(model, tokenizer, model_name, config):
    # Save model
    model.save(f'models/{model_name}.keras')
    
    # Save tokenizer
    with open(f'models/{model_name}_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save configuration
    with open(f'models/{model_name}_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print(f"✅ Model components saved with prefix '{model_name}'")

# Step 10: Enhanced Prediction Pipeline with Visualization
def predict_sentiment(text, model, tokenizer, explanation=True):
    # Preprocess
    cleaned = clean_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=CONFIG['max_len'], padding='post')
    
    # Predict
    prediction = model.predict(padded)[0]
    sentiment_idx = np.argmax(prediction)
    sentiment_label = ['Negative', 'Positive'][sentiment_idx]
    confidence = float(prediction[sentiment_idx])
    
    result = {
        'sentiment': sentiment_label,
        'confidence': confidence,
        'probabilities': {
            'negative': float(prediction[0]),
            'positive': float(prediction[1])
        },
        'text': text,
        'processed_text': cleaned
    }
    
    # Generate visualization if requested
    if explanation:
        try:
            # Simple word contribution visualization (not real SHAP values)
            words = cleaned.split()
            
            # Placeholder for word importance (more sophisticated methods would be used in production)
            word_scores = {}
            
            # Assign scores based on sentiment-related words (simplified)
            pos_indicators = ['good', 'great', 'excellent', 'love', 'awesome', 'amazing', 'happy']
            neg_indicators = ['bad', 'terrible', 'awful', 'hate', 'poor', 'horrible', 'worst']
            
            for word in words:
                if word in pos_indicators:
                    word_scores[word] = 0.7 if sentiment_idx == 1 else -0.3
                elif word in neg_indicators:
                    word_scores[word] = 0.7 if sentiment_idx == 0 else -0.3
                else:
                    word_scores[word] = 0.1  # Neutral
            
            # Create the visualization
            plt.figure(figsize=(12, 4))
            
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
            
            # Save to bytes
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=100)
            plt.close()
            img_buf.seek(0)
            
            result['explanation_image'] = img_buf
            
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
    
    return result

# K-fold Cross Validation
def run_cross_validation(X, y, tokenizer, embedding_matrix=None):
    print("🔄 Running cross-validation...")
    
    # Initialize the cross-validation
    kfold = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    
    # Track metrics
    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1 = []
    
    # For target labels
    fold_targets = np.argmax(y, axis=1)
    
    # Process each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, fold_targets)):
        print(f"\n🔄 Training fold {fold+1}/{CONFIG['k_folds']}")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Calculate class weights
        class_weights = None
        if CONFIG['use_class_weights']:
            class_counts = np.sum(y_train_fold, axis=0)
            class_weights = {
                i: sum(class_counts)/(len(class_counts)*count)
                for i, count in enumerate(class_counts)
            }
        
        # Build model
        model = build_model(tokenizer, embedding_matrix)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate the model
        y_pred = model.predict(X_val_fold)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val_fold, axis=1)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred_classes)
        prec = precision_score(y_true, y_pred_classes, average='weighted')
        rec = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        
        # Store metrics
        cv_accuracy.append(acc)
        cv_precision.append(prec)
        cv_recall.append(rec)
        cv_f1.append(f1)
        
        print(f"Fold {fold+1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Clear the model to free up memory
        tf.keras.backend.clear_session()
    
    # Calculate and print average metrics
    avg_accuracy = np.mean(cv_accuracy)
    avg_precision = np.mean(cv_precision)
    avg_recall = np.mean(cv_recall)
    avg_f1 = np.mean(cv_f1)
    
    std_accuracy = np.std(cv_accuracy)
    std_precision = np.std(cv_precision)
    std_recall = np.std(cv_recall)
    std_f1 = np.std(cv_f1)
    
    print("\n📊 Cross-Validation Results:")
    print(f"Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"Precision: {avg_precision:.4f} (±{std_precision:.4f})")
    print(f"Recall: {avg_recall:.4f} (±{std_recall:.4f})")
    print(f"F1 Score: {avg_f1:.4f} (±{std_f1:.4f})")


def load_and_process_data():
    """Combines data loading, preprocessing, augmentation and preparation"""
    # Step 1: Download data
    df = download_twitter_data()
    
    # Step 2: Preprocess data
    df = preprocess_data(df)
    
    # Step 3: Augment data (if enabled)
    df = augment_dataset(df)
    
    # Step 4: Analyze data (optional)
    df = analyze_data(df)
    
    # Step 5: Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, embedding_matrix = prepare_data(df)
    
    # Combine for cross-validation
    X = np.vstack((X_train, X_val, X_test))
    y = np.vstack((y_train, y_val, y_test))
    
    return X, y, X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, embedding_matrix

def main():
    # Step 1: Load and Preprocess Data
    X, y, X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, embedding_matrix = load_and_process_data()
    
    # Step 2: Class Weights
    class_weights = None
    if CONFIG['use_class_weights']:
        class_counts = np.sum(y_train, axis=0)
        class_weights = {
            i: sum(class_counts)/(len(class_counts)*count)
            for i, count in enumerate(class_counts)
        }

    # Step 3: Build Model
    model = build_model(tokenizer, embedding_matrix)

    # Step 4: Train Model - GPU usage is handled internally via TensorFlow
    history, model_name = train_model(model, X_train, y_train, X_val, y_val, class_weights)

    # Step 5: Evaluate Model
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, tokenizer, model_name)

    # Step 6: Save Components
    save_components(model, tokenizer, model_name, CONFIG)

    # Step 7: Prediction Pipeline
    text = "This movie was the best I have ever seen!"
    result = predict_sentiment(text, model, tokenizer)
    print(result)

    print("Training completed successfully!")

if __name__ == "__main__":
    main()