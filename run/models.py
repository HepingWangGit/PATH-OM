"""
neural_models.py
Contains custom TensorFlow/Keras models and related layers.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Embedding, Flatten, Concatenate, GlobalAveragePooling1D, LayerNormalization, Attention

class AttentionBasedModel(Model):
    def __init__(self, output_dim=289, sequence_len=100, feature_dim=13, num_heads=4, hidden_dim=128):
        """
        A model that reshapes a 1300-dim input to (sequence_len, feature_dim), applies multi-head attention,
        and returns an output vector of size output_dim.
        """
        super(AttentionBasedModel, self).__init__()
        self.sequence_len = sequence_len
        self.feature_dim = feature_dim

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.feature_dim)
        self.ffn = Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(feature_dim)
        ])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.pool = GlobalAveragePooling1D()
        self.out_dense = Dense(output_dim)

    def call(self, inputs, training=False):
        # Reshape from (batch, 1300) to (batch, sequence_len, feature_dim)
        x = tf.reshape(inputs, (-1, self.sequence_len, self.feature_dim))
        attn_output = self.attention(query=x, value=x, key=x, training=training)
        x = self.norm1(x + attn_output, training=training)
        ffn_output = self.ffn(x, training=training)
        x = self.norm2(x + ffn_output, training=training)
        x = self.pool(x)
        outputs = self.out_dense(x)
        return outputs

class Encoder(tf.keras.Model):
    def __init__(self, encoding_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder_layer = Dense(encoding_dim)
        self.leakyRelu = LeakyReLU(negative_slope=0.5)

    def call(self, inputs):
        x = self.encoder_layer(inputs)
        return self.leakyRelu(x)

class Decoder(tf.keras.Model):
    def __init__(self, input_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder_layer = Dense(input_dim)
        self.leakyRelu = LeakyReLU(negative_slope=0.5)

    def call(self, inputs):
        x = self.decoder_layer(inputs)
        return self.leakyRelu(x)

class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, encoder_dim, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(input_dim)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

class TSEquationLayer(layers.Layer):
    def __init__(self, bionet, units, fs, name=None):
        """
        Custom layer that applies a trainable transformation and multiplies by fs.
        """
        super(TSEquationLayer, self).__init__(name=name)
        self.units = units
        self.fs = fs
        self.bionet = bionet

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 trainable=True)
    def call(self, inputs):
        weighted_sum = tf.matmul(inputs, self.W)
        # Multiply elementwise by fs (converted to a tensor)
        fs_tensor = tf.convert_to_tensor(self.fs, dtype=inputs.dtype)
        output = fs_tensor * (inputs + weighted_sum)
        return output

class CustomTSModel(Model):
    def __init__(self, nprots, num_categories, embedding_dim, fs_list, bionet):
        """
        A custom model that combines genomic, experimental, and drug models.
        """
        super(CustomTSModel, self).__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

        self.batch_norm1 = BatchNormalization(name='Batch-Norm-1')
        self.batch_norm2 = BatchNormalization(name='Batch-Norm-2')
        self.batch_norm3 = BatchNormalization(name='Batch-Norm-3')
        
        self.calc_ts = TSEquationLayer(bionet=bionet, units=nprots, fs=fs_list, name='TS-Equation-Layer')
        
        self.genomic_model = Sequential([
            Dense(nprots, activation='relu')
        ], name='Genomic-Model')
        
        self.exp_model = Sequential([
            Dense(nprots, activation='relu')
        ], name='Experiment-Model')
        
        self.drug_model = Sequential([
            Dense(nprots),
            LeakyReLU(negative_slope=0.5)
        ], name='Drug-Model')
        
        self.embedding = Embedding(input_dim=num_categories, output_dim=10, input_length=1, name='Embedding-1')
        self.embedding2 = Embedding(input_dim=4, output_dim=10, input_length=1, name='Embedding-2')
        self.flatten = Flatten()
        self.concat = Concatenate()
        
        self.baseline_process = Sequential([
            Dense(nprots),
            LeakyReLU(negative_slope=0.4)
        ], name='Baseline-Model')
        
        self.mrna_process = Sequential([
            Dense(50, activation='relu')
        ], name='mRNA-Model')
        
        self.response_model = Sequential([
            Dense(300, activation='relu'),
            Dense(nprots),
            LeakyReLU(negative_slope=0.4)
        ], name='Response-Model')
        
        self.cna_dense = Dense(32, activation='relu', name='CNA-Model')
        self.mut1_dense = Dense(32, activation='relu', name='Mutation-Dense-1')
        self.mut2_dense = Dense(32, activation='relu', name='Mutation-Dense-2')
        
        self.drug_dense = Dense(32, activation='relu', name='Drug-Dense')
        self.time_dense = Dense(32, activation='relu', name='Time-Dense')
        self.dose_dense = Dense(32, activation='relu', name='Dose-Dense')
        
    def call(self, inputs):
        # Unpack inputs
        (input_drug, input_time, input_dose, input_dim, input_cna,
         input_mrna, input_mut1, input_mut2, input_baseline) = inputs
        
        baseline_processed = self.baseline_process(input_baseline)
        drug_processed = self.drug_dense(input_drug)
        time_processed = self.time_dense(input_time)
        dose_processed = self.dose_dense(input_dose)
        
        dim_embedded = self.embedding2(input_dim)
        dim_flattened = self.flatten(dim_embedded)
        
        mrna_processed = self.mrna_process(input_mrna)
        cna_processed = self.cna_dense(input_cna)
        mut1_processed = self.mut1_dense(input_mut1)
        mut2_processed = self.mut2_dense(input_mut2)
        
        drug_x = self.drug_model(self.concat([drug_processed, dose_processed]))
        genomic_x = self.genomic_model(self.concat([mrna_processed, cna_processed, mut1_processed, mut2_processed]))
        experiment_x = self.exp_model(self.concat([dim_flattened, time_processed]))
        
        drug_x = self.batch_norm1(drug_x)
        genomic_x = self.batch_norm2(genomic_x)
        experiment_x = self.batch_norm3(experiment_x)
        
        response_model_input = self.concat([genomic_x, drug_x, experiment_x, baseline_processed])
        responses = self.response_model(response_model_input)
        final_output = self.calc_ts(responses)
        return final_output

class WeightedAverageEnsemble:
    def __init__(self, models, avg_check=False):
        self.models = models
        self.weights = None
        self.avg_check = avg_check

    def fit(self, X, y):
        # Train each model
        for name, model in self.models:
            print(name)
            if name in ['rf', 'xgb']:
                X = X
            else:
                scaler_x = StandardScaler()
                X = scaler_x.fit_transform(X)
            model.fit(X, y)
        
        # Calculate weights based on cross-validation performance
        scores = []
        for name, model in self.models:
            model.fit(X, y) #cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            pred = model.predict(X)
            err = np.mean((pred - y)**2)
            scores.append(-err)  # Negative because we want to minimize MSE
        
        # Convert scores to weights (lower MSE gets higher weight)
        self.weights = 1 / np.array(scores)
        self.weights /= np.sum(self.weights)  # Normalize weights

        print(self.weights)
        
        if self.avg_check:
            self.weights = [(1/len(self.models)) for _ in range(len(self.models))]
        
        return self

    def predict(self, X):
        predictions = [model.predict(X) for name, model in self.models]
        output = np.sum(np.array([predictions[i]*self.weights[i] for i in range(len(self.models))]), axis=0)
        return output
# You can similarly add CustomAttentionModel and any additional models here.

class CustomAttentionModel(Model):
   def __init__(self, nprots, num_categories, embedding_dim):
       super(CustomAttentionModel, self).__init__()
       
       self.num_categories = num_categories
       self.embedding_dim = embedding_dim
       self.batch_norm1 = BatchNormalization(name='BatchNorm-1')
       self.batch_norm2 = BatchNormalization(name='BatchNorm-2')
       
       # Define the layers
       self.embedding = Embedding(input_dim=10, output_dim=embedding_dim, input_length=1, name='Embedding-1')
       self.embedding2 = Embedding(input_dim=5, output_dim=10, input_length=1, name='Embedding-2')
       self.flatten = Flatten()
       self.baseline_dense = Dense(64, activation='relu', name='Baseline-Dense')
       self.drug_dense = Dense(64, activation='relu', name='Drug-Dense')
       self.dim_dense = Dense(64, activation='relu', name='Dim-Dense')
       self.mrna_dense = Dense(64, activation='relu', name='mRNA-Dense')
       self.cna_dense = Dense(64, activation='relu', name='CNA-Dense')
       self.mut1_dense = Dense(64, activation='relu', name='Mutation-Dense-1')
       self.mut2_dense = Dense(64, activation='relu', name='Mutation-Dense-2')
       self.time_dense = Dense(64, activation='relu', name='Time-Dense')
       self.dose_dense = Dense(64, activation='relu', name='Dose-Dense')
       self.attention_layer = Attention()
       self.attention_layer2 = Attention()
       self.attention_layer3 = Attention()
       self.attention_layer4 = Attention()
       self.concat_layer = Concatenate()
       self.dense_concat = Sequential([
           Dense(64),
           LeakyReLU(negative_slope=0.5),
       ])
       self.dense_concat2 = Sequential([
           Dense(nprots),
           LeakyReLU(negative_slope=0.5),
       ])
       self.output_layer = Dense(nprots)
       self.leakyRelu = LeakyReLU(negative_slope=0.99)

   def call(self, inputs):
       # Split the inputs; now includes the categorical feature as the last item
       input_drug, input_time, input_dose, input_dim, input_cna, input_mrna, input_mut1, input_mut2, input_baseline = inputs

       #Process time and dose information
       time_processed = self.time_dense(input_time)
       dose_processed = self.dose_dense(input_dose)

       dim_embedded = self.embedding(input_dim)
       dim_flattened = self.flatten(dim_embedded)
       dim_flattened = self.dim_dense(dim_flattened)
       
       # Process vector inputs
       baseline_processed = self.baseline_dense(input_baseline)
       drug_processed = self.drug_dense(input_drug)
       mrna_processed = self.mrna_dense(input_mrna)
       cna_processed = self.cna_dense(input_cna)
       mut1_processed = self.mut1_dense(input_mut1)
       mut2_processed = self.mut2_dense(input_mut2)

       baseline_processed_x = tf.expand_dims(baseline_processed, 1)
       drug_processed_x = tf.expand_dims(drug_processed, 1)
       dose_processed_x = tf.expand_dims(dose_processed, 1)
       mrna_processed_x = tf.expand_dims(mrna_processed, 1)
       cna_processed_x = tf.expand_dims(cna_processed, 1)
       mut1_processed_x = tf.expand_dims(mut1_processed, 1)
       mut2_processed_x = tf.expand_dims(mut2_processed, 1)
       time_processed_x = tf.expand_dims(time_processed, 1)
       dim_flattened_x = tf.expand_dims(dim_flattened, 1)
       
       # Apply attention
       attention_output = self.attention_layer([drug_processed_x, baseline_processed_x, time_processed_x+dim_flattened_x+dose_processed_x])
       attention_output = tf.squeeze(attention_output, 1)

       attention_output2 = self.attention_layer2([cna_processed_x, baseline_processed_x, mrna_processed_x])
       attention_output2 = tf.squeeze(attention_output2, 1)

       attention_output3 = self.attention_layer3([mut1_processed_x + mut2_processed_x, baseline_processed_x, cna_processed_x])
       attention_output3 = tf.squeeze(attention_output3, 1)

       attention_output4 = self.attention_layer4([mrna_processed_x, baseline_processed_x, mut1_processed_x + mut2_processed_x])
       attention_output4 = tf.squeeze(attention_output4, 1)
       
       # Concatenate all outputs including the flattened embedding
       concat_output = self.concat_layer([attention_output, attention_output2, attention_output3, attention_output4])
       concat_output = self.batch_norm1(concat_output)

       # Further processing and output
       dense_output = self.dense_concat(concat_output)

       output = dense_output + baseline_processed
       
       final_output = self.output_layer(output)
       final_output = self.leakyRelu(final_output)
       
       return final_output