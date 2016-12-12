import pic3
import plotting

pic3.two_stream_instability("data_analysis/TS1.hdf5", NT=900)
plotting.plotting("data_analysis/TS1.hdf5")
pic3.two_stream_instability("data_analysis/TS2.hdf5", plasma_frequency=10)
plotting.plotting("data_analysis/TS2.hdf5")
pic3.two_stream_instability("data_analysis/TS3.hdf5", plasma_frequency=10,
                            N_electrons=int(1e5), alpha=0.3)
plotting.plotting("data_analysis/TS3.hdf5")
pic3.two_stream_instability("data_analysis/TS4.hdf5", NT=1000,
                            plasma_frequency=1.1*2**-0.5, N_electrons=int(1e5))
plotting.plotting("data_analysis/TS4.hdf5")
