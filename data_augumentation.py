#!/usr/bin/env python3
"""
Data Augmentation and Class Balancing Module
Handles text augmentation using WordNet synonyms and SMOTE-based balancing
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
import random
import re

class TextAugmentation:
    """Class for text data augmentation to handle class imbalance"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def synonym_replacement(self, sentence, n=1):
        """Replace n words with their synonyms using WordNet"""
        words = sentence.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            
            if num_replaced >= n:
                break
                
        return ' '.join(new_words)
    
    def paraphrase_sentence(self, sentence):
        """Generate multiple paraphrases using synonym replacement"""
        paraphrases = []
        paraphrases.append(sentence)
        paraphrases.append(self.synonym_replacement(sentence, n=1))
        paraphrases.append(self.synonym_replacement(sentence, n=2))
        
        # Remove duplicates and empty strings
        paraphrases = list(set([p for p in paraphrases if p and len(p.strip()) > 0]))
        return paraphrases

class DataBalancer:
    """Class for handling class imbalance through various techniques"""
    
    def __init__(self):
        self.augmenter = TextAugmentation()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess text with enhanced cleaning"""
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        text = re.sub(r'[^\w\s\?]', ' ', text)
        
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 1]
        words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    
    def analyze_class_distribution(self, df, label_column='label'):
        """Analyze and display class distribution"""
        class_counts = df[label_column].value_counts()
        print("\nClass Distribution Analysis:")
        print("-" * 40)
        
        total_samples = len(df)
        for label, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")
        
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
        
        return class_counts, imbalance_ratio
    
    def augment_minority_classes(self, df, target_samples=30, label_column='label', text_column='sentence'):
        """Augment minority classes using text augmentation"""
        print("\nAugmenting minority classes...")
        
        augmented_data = []
        class_counts = df[label_column].value_counts()
        
        for label in class_counts.index:
            current_count = class_counts[label]
            class_data = df[df[label_column] == label].copy()
            
            # Add original samples
            augmented_data.extend(class_data.to_dict('records'))
            
            # If class needs augmentation
            if current_count < target_samples:
                needed_samples = target_samples - current_count
                print(f"  {label}: {current_count} -> {target_samples} samples (+{needed_samples})")
                
                original_sentences = class_data[text_column].tolist()
                generated_count = 0
                
                while generated_count < needed_samples:
                    for sentence in original_sentences:
                        if generated_count >= needed_samples:
                            break
                            
                        paraphrases = self.augmenter.paraphrase_sentence(sentence)
                        
                        for paraphrase in paraphrases[1:]:  # Skip original
                            if generated_count >= needed_samples:
                                break
                                
                            augmented_data.append({
                                text_column: paraphrase,
                                label_column: label
                            })
                            generated_count += 1
        
        augmented_df = pd.DataFrame(augmented_data)
        print(f"\nDataset augmented: {len(df)} -> {len(augmented_df)} samples")
        
        return augmented_df
    
    def apply_smote_balancing(self, X_vectorized, y, random_state=42):
        """Apply SMOTE for additional balancing on vectorized data"""
        print("\nApplying SMOTE for additional balancing...")
        
        # Check minimum class size for k_neighbors parameter
        min_class_size = min(np.bincount(y))
        k_neighbors = min(5, min_class_size - 1)
        
        if k_neighbors < 1:
            print("Warning: Class sizes too small for SMOTE. Skipping SMOTE.")
            return X_vectorized, y
        
        try:
            smote_tomek = SMOTETomek(
                smote=SMOTE(random_state=random_state, k_neighbors=k_neighbors),
                tomek=TomekLinks(),
                random_state=random_state
            )
            X_balanced, y_balanced = smote_tomek.fit_resample(X_vectorized, y)
            
            original_shape = X_vectorized.shape[0]
            balanced_shape = X_balanced.shape[0]
            
            print(f"SMOTE applied: {original_shape} -> {balanced_shape} samples")
            print(f"New class distribution: {np.bincount(y_balanced)}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
            print("Returning original data without SMOTE balancing.")
            return X_vectorized, y
    
    def get_balanced_data(self, df, use_augmentation=True, use_smote=True, 
                         target_samples=30, label_column='label', text_column='sentence'):
        """Complete data balancing pipeline"""
        print("="*60)
        print("DATA BALANCING PIPELINE")
        print("="*60)
        
        # Analyze original distribution
        original_counts, original_ratio = self.analyze_class_distribution(df, label_column)
        
        balanced_df = df.copy()
        
        # Apply text augmentation if requested
        if use_augmentation:
            balanced_df = self.augment_minority_classes(
                balanced_df, target_samples, label_column, text_column
            )
            
            # Analyze after augmentation
            print("\nAfter Text Augmentation:")
            self.analyze_class_distribution(balanced_df, label_column)
        
        return balanced_df
    
    def create_class_weights(self, y):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced', 
            classes=classes, 
            y=y
        )
        
        class_weight_dict = dict(zip(classes, class_weights))
        
        print("\nCalculated Class Weights:")
        for class_id, weight in class_weight_dict.items():
            print(f"  Class {class_id}: {weight:.3f}")
            
        return class_weight_dict

def setup_nltk():
    """Download required NLTK data"""
    required_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for item in required_downloads:
        try:
            nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else 
                          f'corpora/{item}' if item in ['stopwords', 'wordnet'] else
                          f'taggers/{item}')
        except LookupError:
            print(f"Downloading NLTK {item}...")
            nltk.download(item, quiet=True)

# Utility functions for easy import
def quick_augment_data(df, target_samples=30):
    """Quick function to augment data"""
    balancer = DataBalancer()
    return balancer.get_balanced_data(df, target_samples=target_samples)

def quick_smote_balance(X, y):
    """Quick function to apply SMOTE"""
    balancer = DataBalancer()
    return balancer.apply_smote_balancing(X, y)

if __name__ == "__main__":
    # Example usage
    setup_nltk()
    
    # Demo with sample data
    sample_data = {
        'sentence': [
            'I want EMI option',
            'Cancel my order',
            'What is warranty?',
            'I want EMI please',
            'Check delivery',
        ],
        'label': ['emi', 'cancel', 'warranty', 'emi', 'delivery']
    }
    
    df = pd.DataFrame(sample_data)
    balancer = DataBalancer()
    
    print("DEMO: Data Augmentation Module")
    balanced_df = balancer.get_balanced_data(df, target_samples=10)
    print(f"\nFinal dataset size: {len(balanced_df)}")