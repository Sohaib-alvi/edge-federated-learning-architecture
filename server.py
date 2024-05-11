# import socketio
# import eventlet
# import numpy as np

# import msgpack
# import msgpack_numpy as mn

# import json
# import matplotlib.pyplot as plt

# import tensorflow as tf
# from keras.applications.densenet import DenseNet121
# from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
# from keras.models import Sequential
# from keras.optimizers import Adam
# from tensorflow.compat.v1.keras import backend as K
# import codecs
# import pickle

# sio = socketio.Server()
# app = socketio.WSGIApp(sio)

# def obj_to_pickle_string(x):
#     return codecs.encode(pickle.dumps(x), "base64").decode()
#     # return msgpack.packb(x, default=msgpack_numpy.encode)
#     # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

# def pickle_string_to_obj(s):
#     return pickle.loads(codecs.decode(s.encode(), "base64"))
#     # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)

# def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
#     """
#     Return weighted loss function given negative weights and positive weights.

#     Args:
#       pos_weights (np.array): array of positive weights for each class, size (num_classes)
#       neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
#     Returns:
#       weighted_loss (function): weighted loss function
#     """
#     def weighted_loss(y_true, y_pred):
#         """
#         Return weighted loss value. 

#         Args:
#             y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
#             y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
#         Returns:
#             loss (Tensor): overall scalar loss summed across all classes
#         """
#         # initialize loss to zero
#         loss = 0.0

#         for i in range(len(pos_weights)):
#             # for each class, add average weighted loss for that class 
#             loss += -(K.mean( pos_weights[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon) + \
#                                 neg_weights[i] * (1 - y_true[:,i]) * K.log(1 - y_pred[:,i] + epsilon), axis = 0))
#         return loss
    
#     return weighted_loss
# def json_serialize(weights):
#     serialized_weights = lambda a: [i.tolist() for i in a]
#     return serialized_weights(weights)

# def json_deserialize(weights):
#     deserialized_weights = lambda a: [np.array(i) for i in a]
#     return deserialized_weights(weights)


# class Flserver(socketio.Namespace):


    

#     def __init__(self):

#         super(Flserver, self).__init__()
#         self.clients = np.array([])
#         self.K = 0                                  
#         self.C = 7/10
#         self.current_weights = []
#         self.current_round = 1
#         self.max_round = 4
#         self.scores = np.array([])
        
#     def build_model(self, pos_weights, neg_weights):

#         model = Sequential()
#         model.add(Conv2D(60, (5,5), input_shape=(32,32,1), activation="relu"))
#         model.add(Conv2D(60, (5,5), activation="relu"))
#         model.add(MaxPooling2D(pool_size=(2,2)))

#         model.add(Conv2D(30, (3, 3), activation='relu'))
#         model.add(Conv2D(30, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2,2)))
#         #model.add(Dropout(0.5))

#         model.add(Flatten())
#         model.add(Dense(500, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(2, activation='softmax'))
#         #Compile model
        
#         model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#         return model

#     def on_connect(self, sid, environ):
        
#         print(f"Client connected: {sid}")

#     def on_model_request(self, sid, data):
#         print("server preparing the model")
#         pos_weights = msgpack.unpackb(data["pos"], object_hook=mn.decode)
#         neg_weights = msgpack.unpackb(data["neg"], object_hook=mn.decode)
#         self.model = self.build_model(pos_weights, neg_weights)
        
#         print(self.model.summary())

#         sio.emit("sending_model", {"model": self.model.to_json(), "n_round":1})


#     def on_disconnect(self, sid):
        
#         print(f"Client disconnected: {sid}")

#     def on_complete_round(self, sid, data):
        
#         print("Round completed")

#         self.clients = np.append(self.clients, data)
#         self.K = self.clients.shape[0]
#         m = max(self.C*self.K, 1)
#         self.clients_updaters = np.random.choice(self.clients, int(m))
#         n_k = np.array([i["n_train"] for i in self.clients_updaters if i["n_round"]==r])
#         n_k_test = np.array([i["n_test"] for i in self.clients_updaters])
#         n_test = np.sum(n_k_test)
#         n = np.sum(n_k)
#         current_weights = np.array([pickle_string_to_obj(i["weights"]) for i in self.clients_updaters])
#         current_scores = np.array([i["scores"] for i in self.clients_updaters])
#         new_scores = n_k_test*np.sum(current_scores, axis = 0)/n_test
#         new_weights = n_k*np.sum(current_weights, axis = 0)/n
#         self.current_weights = new_weights
#         self.scores = np.append(self.scores, new_scores)
        
#         if self.n_round <= self.max_round:
            
#             n_round = r + 1
#             print("Next round soon...")
#             print(len(new_weights))
#             print(new_weights[0].shape)
#             server_packet = {"model": obj_to_pickle_string(new_weights), "n_round": self.n_round}
#             return server_packet

#         else:

#             fig, axs = plt.subplots(2, 1)
#             axs[0].plot([i+1 for i in range(self.max_round)], self.scores[:,0])
#             axs[0].set_xlabel('Round')
#             axs[0].set_ylabel('Agg Test Score')

#             axs[1].plot([i+1 for i in range(self.max_round)], self.scores[:,1])
#             axs[1].set_xlabel('Round')
#             axs[1].set_ylabel('Agg Test Accuracy')

#             fig.tight_layout()
#             plt.show()

#             print("model updated")

        

# sio.register_namespace(Flserver())

# if __name__ == "__main__":
#     eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
import socketio
import eventlet
import numpy as np

import msgpack
import msgpack_numpy as mn

import json
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.compat.v1.keras import backend as K
import codecs
import pickle

def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += -(K.mean( pos_weights[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon) + \
                                neg_weights[i] * (1 - y_true[:,i]) * K.log(1 - y_pred[:,i] + epsilon), axis = 0))
        return loss
    
    return weighted_loss
def json_serialize(weights):
    serialized_weights = lambda a: [i.tolist() for i in a]
    return serialized_weights(weights)

def json_deserialize(weights):
    deserialized_weights = lambda a: [np.array(i) for i in a]
    return deserialized_weights(weights)


class Flserver():


    

    def __init__(self):

        self.clients = np.array([])
        self.K = 0                                  
        self.C = 7/10
        self.current_weights = []
        self.n_round = 1
        self.max_round = 4
        self.scores = np.array([[0, 0]])
        self.sio = socketio.Server()
        self.app = socketio.WSGIApp(self.sio)
        self.register_handlers()
        
        

    def build_model(self):

        model = Sequential()
        model.add(Conv2D(60, (5,5), input_shape=(32,32,1), activation="relu"))
        model.add(Conv2D(60, (5,5), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(30, (3, 3), activation='relu'))
        model.add(Conv2D(30, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        #Compile model
        
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model

    def register_handlers(self):

        @self.sio.event
        def connect(sid, environ):
            
            print(f"Client connected: {sid}")

        @self.sio.event
        def model_request(sid, data):
            print("server preparing the model")
            """ pos_weights = msgpack.unpackb(data["pos"], object_hook=mn.decode)
            neg_weights = msgpack.unpackb(data["neg"], object_hook=mn.decode) """
            self.model = self.build_model()
            
            print(self.model.summary())

            self.sio.emit("sending_model", {"model": self.model.to_json(), "n_round":1})

        @self.sio.event
        def disconnect(sid):
            
            print(f"Client disconnected: {sid}")

        @self.sio.event
        def complete_round(sid, data):
            
            print("Round completed")

            self.clients = np.append(self.clients, data)
            clients = [i for i in self.clients if i["n_round"]==self.n_round]
            self.K = len(clients)
        
            m = max(self.C*self.K, 1)
            self.clients_updaters = np.random.choice(clients, int(m))
            n_k = np.array([i["n_train"] for i in self.clients_updaters ])
            n_k_test = np.array([i["n_test"] for i in self.clients_updaters])
            n_test = np.sum(n_k_test)
            n = np.sum(n_k)
            current_weights = np.array([pickle_string_to_obj(i["weights"]) for i in self.clients_updaters])
            current_scores = np.array([i["scores"] for i in self.clients_updaters])
            new_scores = n_k_test*np.sum(current_scores, axis = 0)/n_test
            new_weights = n_k*np.sum(current_weights, axis = 0)/n
            self.current_weights = new_weights
            self.scores = np.append(self.scores, [new_scores], axis = 0)
            self.n_round += 1
            if self.n_round <= self.max_round:
                
                
                print("Next round soon...")
                print(len(new_weights))
                print(new_weights[0].shape)
                server_packet = {"model": obj_to_pickle_string(new_weights), "n_round": self.n_round}
                return server_packet

            else:
                print(self.scores)
                fig, axs = plt.subplots(2, 1)
                axs[0].plot([i+1 for i in range(self.max_round)], self.scores[1:,0])
                axs[0].set_xlabel('Round')
                axs[0].set_ylabel('Agg Test Score')

                axs[1].plot([i+1 for i in range(self.max_round)], self.scores[1:,1])
                axs[1].set_xlabel('Round')
                axs[1].set_ylabel('Agg Test Accuracy')

                fig.tight_layout()
                plt.show()

                print("model updated")

    def start(self):
        eventlet.wsgi.server(eventlet.listen(('', 5000)), self.app)
        



if __name__ == "__main__":
    flserver = Flserver()
    flserver.start()
