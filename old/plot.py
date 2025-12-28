from matplotlib import pyplot as plt
import pandas as pd
import stumpy
import numpy as np
import os
from matplotlib.patches import Rectangle
import datetime
import pytz
import signal
from multiprocessing import Pool
from numpy.lib.stride_tricks import as_strided

signal.signal(signal.SIGINT, signal.SIG_DFL)
window = 6
extra = 10
stock_data = pd.read_csv("alpaca_historical_data_SPY_20150101_20250505.csv")
wap_np = stock_data["WAP"].to_numpy()
wap_np = wap_np

# if not os.path.exists('stump.npy'):
#     mp = stumpy.stump(stock_data['WAP'], window)
#     # np.save('stump', mp)
# else:
#     # mp = np.load('stump.npy', allow_pickle=True)
#     pass

# plt.plot(mp[:, 0], label='Distance')
# plt.show()

# print(mp[:, 0])
# second_order_stump = stumpy.stump(list(mp[:, 0]), window)
# third_order_stump = stumpy.stump(list(second_order_stump[:, 0]), 640)


# for motif in np.argsort(mp[:, 0]):
# for motif in mp[:, 0]:
# for motif in range(0, len(mp), 50):
#     nn = mp[motif, 1]
#     fig, axs = plt.subplots(4)
#     rect = Rectangle((motif, 0), window, 1000, facecolor='lightgrey')
#     axs[0].plot(stock_data['WAP'].values, label='WAP')
#     axs[0].add_patch(rect)
#     rect = Rectangle((nn, 0), window, 1000, facecolor='lightgrey')
#     axs[0].add_patch(rect)
#     axs[1].plot(wap_np[motif:motif + window + extra], label=f'Motif {motif}')
#     axs[1].plot(wap_np[nn:nn + window + extra], label=f'NN {nn}')
#     axs[1].set_ylabel(f'distance {mp[motif, 0]}')
#     axs[1].legend()
#     axs[2].plot(mp[:, 0], label='Distance')
#     axs[3].plot(second_order_stump[:, 0], label='Seond Order distance')
#     figManager = plt.get_current_fig_manager()
#     figManager.full_screen_toggle()
#     plt.show()


class TrainingEvalData:
    def __init__(self, data: np.ndarray, training_data_size: int):
        self.data = data
        self.training_data = data[:training_data_size]
        self.eval_data = data[training_data_size:]


def simulator_helper(training_eval_data: TrainingEvalData):
    stream = stumpy.stumpi(training_eval_data.training_data, window, egress=False)
    gains = 0
    for current, wap in enumerate(
        training_eval_data.eval_data, start=training_eval_data.training_data.shape[0]
    ):
        if current == len(training_eval_data.data) - 1:
            break
        stream.update(wap)
        idx = stream.I_[-1]
        distance = stream.P_[-1]
        # print(f"distance is {distance}")
        # print(f"current is {current}")
        # print(idx)
        # print(batch[idx], batch[idx + 1])
        if distance > 0:
            if training_eval_data.data[idx] > training_eval_data.data[idx + 1]:
                pass
                # gains += 10_000 * ((training_eval_data.data[current] - training_eval_data.data[current + 1]) / training_eval_data.data[current])
            else:
                gains += 10_000 * (
                    (
                        training_eval_data.data[current + 1]
                        - training_eval_data.data[current]
                    )
                    / training_eval_data.data[current]
                )
                # pass
    print(f"subgains is {gains}")
    return gains


def split_into_batches(
    data, training_data_size: int, eval_data_size: int
) -> list[TrainingEvalData]:
    result = []
    current = 0
    while current < len(data):
        training_eval_data = data[
            current : current + training_data_size + eval_data_size
        ]
        result.append(TrainingEvalData(training_eval_data, training_data_size))
        current += eval_data_size
    return result


def simulate():
    print("starting simulation")
    # wap_np_batches = np.array_split(wap_np, len(wap_np) // 5000)
    training_data_size = int(10 * 6.5 * 12)
    eval_data_size = int(6.5 * 12)
    training_datas = split_into_batches(wap_np, training_data_size, eval_data_size)
    gains = 0
    gains_list = []
    for day, training_data in enumerate(training_datas):
        print(f"day {day}")
        simulation_result = simulator_helper(training_data)
        gains += simulation_result
        print(f"gains is {gains}")
        gains_list.append(gains)
    print(f"Final gains is {gains}")
    plt.plot(gains_list, label="Gains")
    plt.plot(wap_np, label="WAP")
    plt.title("Gains")
    plt.xlabel("Time")
    plt.show()


simulate()

# plt.plot(stock_data['WAP'].values)
# plt.show()
