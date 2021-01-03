from pathlib import Path
import sqlite3
import numpy as np


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


class SqliteFetcher:
    def __init__(self, db_path):

        self._path = str(db_path)
        self._tables = None
        self._len = None
        self._rows = None
        self._event_lengths_key = "split_in_ice_pulses_event_length"
        self._max_events_per_query = 50000
        # self._fetch_query_seq = 'SELECT {features} FROM {table} WHERE event '\

        self._fetch_query_seq = (
            "SELECT {features} FROM {table} WHERE event_no " "IN ({events})"
        )
        self._fetch_query = (
            "SELECT {features} FROM {table} WHERE event_no " "IN ({events})"
        )
        self._read_query = (
            "SELECT {feature} FROM {table} WHERE {primary_key} IN ({nums})"
        )

    def __len__(self):
        return self._len

    @property
    def ids(self):

        with sqlite3.connect(self._path) as db:
            cursor = db.cursor()
            query = "SELECT event_no FROM meta"
            cursor.execute(query)

            event_ids = [str(e[0]) for e in cursor.fetchall()]

        return event_ids

    @property
    def rows(self):

        with sqlite3.connect(self._path) as db:
            cursor = db.cursor()
            query = "SELECT row FROM sequential"
            cursor.execute(query)

            rows = sorted([str(e[0]) for e in cursor.fetchall()])

        return rows

    @property
    def length(self):

        if not self._len:

            with sqlite3.connect(self._path) as db:
                cursor = db.cursor()
                query = "SELECT event_no FROM meta"
                cursor.execute(query)

                event_nums = [e[0] for e in cursor.fetchall()]
            self._len = len(event_nums)

        return self._len

    @property
    def n_rows(self):

        if not self._rows:

            with sqlite3.connect(self._path) as db:
                cursor = db.cursor()
                query = "SELECT COUNT(*) FROM sequential"
                cursor.execute(query)

                row_nums = cursor.fetchall()
            self._rows = row_nums[0][0]

        return self._rows

    @property
    def tables(self):

        if not self._tables:

            with sqlite3.connect(self._path) as db:
                cursor = db.cursor()

                # Get table-names
                query = 'SELECT name FROM sqlite_master WHERE type = "table"'
                cursor.execute(query)
                tables_data = {entry[0]: {} for entry in cursor.fetchall()}

                # Loop over all columns and fetch their info
                for name in tables_data:
                    query = "PRAGMA TABLE_INFO({tablename})".format(tablename=name)

                    cursor.execute(query)
                    col_data = cursor.fetchall()

                    tables_data[name] = {
                        e[1]: {"type": e[2], "index": e[0],} for e in col_data
                    }

            self._tables = tables_data

        return self._tables

    def _fetch(self, ids, *queries):

        fetched = ()
        with sqlite3.connect(self._path) as db:
            for query in queries:
                cursor = db.cursor()
                cursor.execute(query, ids)
                fetched = fetched + (cursor.fetchall(),)

        return fetched

    def _make_dict(
        self,
        events,
        names_scalar,
        fetched_scalar,
        names_sequential,
        fetched_sequential,
        names_meta,
        fetched_meta,
        event_lengths,
    ):

        # get the from- and to-indices of each event.
        cumsum = np.append([0], np.cumsum([entry[0] for entry in event_lengths]))
        all_from = cumsum[:-1]
        all_to = cumsum[1:]

        # Create dictionary. First level is event
        data_dict = {}
        for i_event, event in enumerate(events):

            # Second level is data
            data_dict[event] = {}

            # order the data from fetched_scalar
            from_, to_ = all_from[i_event], all_to[i_event]
            for i_name, name in enumerate(names_sequential):
                data = [entry[i_name] for entry in fetched_sequential[from_:to_]]
                data_dict[event][name] = np.array(data)

            # Do the same for scalar data
            for i_name, name in enumerate(names_scalar):
                data_dict[event][name] = fetched_scalar[i_event][i_name]

            # .. And finally meta
            for i_name, name in enumerate(names_meta):
                data_dict[event][name] = fetched_meta[i_event][i_name]

        return data_dict

    def fetch_features(
        self,
        all_events=[],
        scalar_features=[],
        seq_features=[],
        meta_features=[],
        reg_type=None,
    ):

        # Connect to DB and set cursor
        with sqlite3.connect(self._path) as db:
            cursor = db.cursor()
            n_events = len(all_events)
            # Ensure some events are passed
            if n_events == 0:
                raise ValueError("NO EVENTS PASSED TO SQLFETCHER")

            # If events are not strings, raise error
            if not isinstance(all_events[0], str):
                raise ValueError("SqliteFetcher: IDs must be strings!")

            if "event_no" in scalar_features or "event_no" in meta_features:
                raise KeyError("event_no cannot be requested!")

            # load over several rounds
            n_chunks = n_events // self._max_events_per_query
            chunks = np.array_split(all_events, max(1, n_chunks))

            base_query = (
                "SELECT {features} FROM {table} WHERE event_no " "IN ({events})"
            )
            base_seq_query = (
                "SELECT {features} FROM {table} WHERE event_no " "IN ({events})"
            )

            fetched_scalar, fetched_sequential, fetched_meta = [], [], []

            # Process chunks
            all_dicted_data = {}
            for events in chunks:
                # Write query for scalar table and fetch all matching rows
                if len(scalar_features) > 0:
                    query = base_query.format(
                        features=", ".join(scalar_features),
                        table="scalar",
                        events=", ".join(["?"] * len(events)),
                    )

                    cursor.execute(query, events)
                    fetched_scalar = cursor.fetchall()

                # Write query for sequential table and fetch all matching rows
                if len(seq_features) > 0:
                    query = base_seq_query.format(
                        features=", ".join(seq_features),
                        table="sequential",
                        events=", ".join(["?"] * len(events)),
                    )
                    cursor.execute(query, events)
                    fetched_sequential = cursor.fetchall()

                # Write query for meta table and fetch all matching rows
                if len(meta_features) > 0:
                    query = base_query.format(
                        features=", ".join(meta_features),
                        table="meta",
                        events=", ".join(["?"] * len(events)),
                    )
                    cursor.execute(query, events)
                    fetched_meta = cursor.fetchall()

                # Finally, fetch event lengths as they are needed for making
                # sequential dictionary
                query = base_query.format(
                    features=self._event_lengths_key,
                    table="meta",
                    events=", ".join(["?"] * len(events)),
                )
                cursor.execute(query, events)
                event_lengths = cursor.fetchall()

                # Put in a dictionary and update all_dicted_ata
                dicted_data = self._make_dict(
                    events,
                    scalar_features,
                    fetched_scalar,
                    seq_features,
                    fetched_sequential,
                    meta_features,
                    fetched_meta,
                    event_lengths,
                )
                all_dicted_data.update(dicted_data)

            return all_dicted_data

    def make_batch(
        self,
        ids=[],
        scalars=[],
        seqs=[],
        targets=[],
        weights=[],
        mask=[],
        reg_type=None,
    ):

        n_events = len(ids)
        # Ensure some events are passed
        if n_events == 0:
            raise ValueError("NO EVENTS PASSED TO SQLFETCHER")

        # If events are not strings, convert them
        if not isinstance(ids[0], str):
            raise ValueError("Events must be IDs as strings")

        if weights == []:
            NO_WEIGHTS = True
        else:
            NO_WEIGHTS = False

        # Prepare single-number queries
        scalar_cols = ["scalar." + e for e in scalars]
        target_cols = ["scalar." + e for e in targets]
        lengths_key = ["meta.split_in_ice_pulses_event_length"]
        weights_col = ["scalar." + e for e in weights]

        all_single_val_feats = scalar_cols + target_cols + weights_col + lengths_key
        singles_query = "SELECT {features} FROM scalar INNER JOIN meta ON scalar.event_no=meta.event_no WHERE scalar.event_no IN ({events})".format(
            features=", ".join(all_single_val_feats), events=", ".join(["?"] * n_events)
        )

        # Prepare sequences-query
        all_seq_cols_feats = seqs + mask
        seq_query = "SELECT {features} FROM sequential WHERE event_no IN ({events})".format(
            features=", ".join(all_seq_cols_feats), events=", ".join(["?"] * n_events)
        )

        # Make fetch
        singles, sequences = self._fetch(ids, singles_query, seq_query)

        # prepare the batch
        batch = []

        # get the from- and to-indices of each event.
        lengths = np.array([entry[-1] for entry in singles])
        cumsum = np.append([0], np.cumsum(lengths))
        all_from = cumsum[:-1]
        all_to = cumsum[1:]

        n_scalars = len(scalars)
        n_targets = len(target_cols)
        n_seqs = len(seqs)
        for i_event in range(n_events):

            from_, to_ = all_from[i_event], all_to[i_event]
            masked_indices = np.array([e[-1] for e in sequences[from_:to_]], dtype=bool)
            n_doms = np.sum(masked_indices)
            seq_arr = np.zeros((n_seqs, n_doms))
            seq_dict = {}
            for i_var, var in enumerate(seqs):
                seq_arr[i_var, :] = np.array(
                    [dom[i_var] for dom in sequences[from_:to_]]
                )[masked_indices]
                seq_dict[var] = np.array([dom[i_var] for dom in sequences[from_:to_]])[
                    masked_indices
                ]

            # Add to list of events
            batch.append(seq_dict)

        return batch

    def read(self, table, feature, primary_key, nums):
        # If events are not strings, convert them
        if not isinstance(nums[0], str):
            raise ValueError("Events must be IDs as strings")
        with sqlite3.connect(self._path) as db:
            cursor = db.cursor()
            query = self._read_query.format(
                # feature=', '.join([feature, primary_key]),
                feature=feature,
                table=table,
                primary_key=primary_key,
                nums=", ".join(["?"] * len(nums)),
            )

            cursor.execute(query, nums)
            fetched = cursor.fetchall()
            converted = np.array([e[0] for e in fetched])
            # ids = np.array([
            #     e[1] for e in fetched
            # ])
        return converted  # , ids

