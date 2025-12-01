import numpy as np


def get_sample_ids(query, horizon, down_sample_steps, backwards=False, closed=False):
    """
    Get the sample ids for the down-sampling of the data.
    :param query: (Q,) The query id(s).
    :param horizon: Integer scalar. The horizon H of the data.
    :param down_sample_steps:Integer scalar. The number of steps to down-sample.
    :param backwards: Boolean. Whether to sample backwards from the query point.
    :param closed: Boolean. Whether to include the closed end.

    :return: The sampled ids (Q, H)
    """
    if isinstance(query, list):
        query = np.array(query)

    if closed:
        sample_horizon = horizon + 1
    else:
        sample_horizon = horizon

    local_id = np.arange(sample_horizon) * down_sample_steps

    if not backwards:
        return [query_i + local_id for query_i in query]
    else:
        local_id = local_id[::-1]
        return [query_i - local_id for query_i in query]


def get_samples(
    raw_data, query, horizon, down_sample_steps, backwards=False, closed=False
):
    """
    Get the sample ids for the down-sampling of the data.
    :param raw_data: (T, D) The raw data. D could have multiple dimnsions.
    For other parameters, see get_sample_ids.

    :return: The sampled data (Q, H, ...).
    """
    return raw_data[
        get_sample_ids(query, horizon, down_sample_steps, backwards, closed)
    ]


def get_dense_query_points_in_horizon(
    sparse_total_steps_per_horizon,
    dense_action_horizon,
    dense_action_down_sample_steps,
    delta_steps,
):
    """
    Get the dense query points in the horizon. The queries, which are indices in the original
    raw array, is also used as time steps of dense queries.
    Adjacent query points are delta_steps apart.

    Example when delta_steps = 3, sparse total steps = 20, dense total steps = 6:
        sparse raw steps:      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
        dense queries:         * - - * - - * - - * -  -  *  -  -
        dense horizons         0 1 2 3 4 5
                                     0 1 2 3 4 5
                                           0 1 2 3 4  5
                                                 0 1  2  3  4  5
                                                         0  1  2  3  4  5
        num of queries = 5, since the 6th query is out of the horizon.

    :param delta_steps: Integer scalar. The number of steps between two adjacent query points.
    """

    sparse_raw_steps = sparse_total_steps_per_horizon
    dense_raw_steps = dense_action_horizon * dense_action_down_sample_steps
    num_of_queries = (sparse_raw_steps - dense_raw_steps) // delta_steps + 1
    queries = np.arange(num_of_queries) * delta_steps
    return queries


def test():
    # test
    raw_data = 10 * np.arange(20)
    query = [10, 12]
    horizon = 3
    down_sample_steps = 2

    print("== Test get_samples ==")
    s = get_samples(
        raw_data, query, horizon, down_sample_steps, backwards=True, closed=False
    )
    print("raw_data:", raw_data)
    print("query:", query)
    print("horizon:", horizon)
    print("down_sample_steps:", down_sample_steps)

    print(s)

    print("== Test get_dense_query_points_in_horizon ==")
    sparse_action_horizon = 5
    sparse_action_down_sample_steps = 4
    dense_action_horizon = 2
    dense_action_down_sample_steps = 3
    delta_steps = 3
    queries = get_dense_query_points_in_horizon(
        sparse_action_horizon,
        sparse_action_down_sample_steps,
        dense_action_horizon,
        dense_action_down_sample_steps,
        delta_steps,
    )
    print(queries)


if __name__ == "__main__":
    test()
