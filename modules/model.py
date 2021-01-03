from pathlib import Path
from typing import List, Dict
import sklearn
import pickle
import torch
import json
import numpy as np
from modules.model_builder import MakeModel


class IceGRU:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        self._model_path = model_path
        self.device = device
        self.transformers = self._load_transformers()
        self.model = self._load_model(self._model_path)
        self._n_seqs = len(self.__seq_vars)

    def predict(self, batch: List[Dict[str, np.ndarray]]) -> List[Dict[str, float]]:
        """Calculates predictions on a batch of data.

        The batch of data must be a list of dictionaries, where each dictionary contains the key-value pairs 
            - dom_x: a numpy-array of the x-coordinates of the event
            - dom_y: a numpy-array of the y-coordinates of the event
            - dom_z: a numpy-array of the z-coordinates of the event
            - dom_time: a numpy-array of the time-coordinates of the event
            - dom_charge: a numpy-array of the charge-values of the event
            - dom_atwd: a numpy-array with digitizer indicators (integers)
            - dom_pulse_width: a numpy-array of pulse widths of the event.

        The event is expected to be time-ordered.
        
        Args:
            batch (List[Dict[str, np.ndarray]]): A batch of event as described above

        Returns:
            List[Dict[str, float]]: Predictions for events
        """

        batch_list_transformed = self._dicts_to_arrays(self._transform_batch(batch))
        batch_packed_sequence, sequence_lengths, new_order = self._pad_sequence(
            batch_list_transformed
        )
        batch_packed = (batch_packed_sequence, sequence_lengths)
        prediction_transformed = self._predict(batch_packed)
        prediction = self._array_to_dicts(
            self._inverse_transform(prediction_transformed.numpy())
        )
        prediction_reordered = [
            e[0] for e in sorted(zip(prediction, new_order), key=lambda x: x[1])
        ]

        return prediction_reordered

    def _dict_to_array(self, event):
        n_doms = len(event[self.__seq_vars[0]])
        seq_arr = np.zeros((self._n_seqs, n_doms))
        for i_var, var in enumerate(self.__seq_vars):
            seq_arr[i_var, :] = event[var]
        return seq_arr

    def _dicts_to_arrays(self, batch):
        for i_event, event in enumerate(batch):
            batch[i_event] = self._dict_to_array(event)
        return batch

    def _inverse_transform(self, pred_array):

        for i_var, var in enumerate(self.__targets):
            transformer = self.transformers.get(var)
            pred = pred_array[:, i_var]
            if transformer:
                inv_transformed_pred = transformer.inverse_transform(
                    pred.reshape(-1, 1)
                ).reshape(-1)
            pred_array[:, i_var] = inv_transformed_pred if transformer else pred

        return pred_array

    def _load_model(self, path):

        with open(Path.joinpath(path, "architecture_pars.json"), "r") as f:
            arch_pars = json.load(f)
        model = MakeModel(arch_pars)
        p = Path.joinpath(path, "model_weights.pth")
        model.load_state_dict(torch.load(p, map_location="cpu"))
        model.to(self.device)

        return model

    def _load_transformers(self):
        with open(self.__transformers_path, "rb") as f:
            transformers = pickle.load(f)

        return transformers

    def _pad_sequence(self, batch):
        indexed_batch = [(entry, i_entry) for i_entry, entry in enumerate(batch)]
        sorted_batch = sorted(indexed_batch, key=lambda x: x[0].shape[1], reverse=True)
        sequences = [torch.tensor(np.transpose(x[0])) for x in sorted_batch]
        indices = [x[1] for x in sorted_batch]
        sequence_lengths = torch.LongTensor([len(x) for x in sequences])
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        return sequences_padded.float(), sequence_lengths, indices

    def _predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(batch)

        return y_pred

    def _array_to_dict(self, prediction):

        prediction_dict = {}
        for i_var, var in enumerate(self.__targets):
            prediction_dict[var] = prediction[i_var]

        return prediction_dict

    def _array_to_dicts(self, prediction_array):
        prediction_dicts = []

        for i_event in range(prediction_array.shape[0]):
            prediction = prediction_array[i_event, :]
            prediction_dicts.append(self._array_to_dict(prediction))

        return prediction_dicts

    def _transform_batch(self, batch):
        for i_event, event in enumerate(batch):
            batch[i_event] = self._transform_event(event)

        return batch

    def _transform_event(self, event):
        for var in self.__seq_vars:
            transformer = self.transformers.get(var)
            data = event[var]

            if transformer:
                transformed_data = transformer.transform(data.reshape(-1, 1)).reshape(
                    -1
                )

            event[var] = transformed_data if transformer else data

        return event

    @property
    def __transformers_path(self):
        return Path.joinpath(self._model_path, "transformers.pickle")

    @property
    def __seq_vars(self):
        seq_vars = [
            "dom_charge",
            "dom_x",
            "dom_y",
            "dom_z",
            "dom_time",
            "dom_atwd",
            "dom_pulse_width",
        ]
        return seq_vars

    @property
    def __targets(self):
        if self._model_path.stem == "direction":
            target_keys = [
                "true_primary_direction_x",
                "true_primary_direction_y",
                "true_primary_direction_z",
            ]
        else:
            target_keys = [
                "true_primary_energy",
                "true_primary_position_x",
                "true_primary_position_y",
                "true_primary_position_z",
                "true_primary_time",
                "true_primary_direction_x",
                "true_primary_direction_y",
                "true_primary_direction_z",
            ]

        return target_keys

