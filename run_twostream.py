import pic3
import plotting


pic3.two_stream_instability("data_analysis/TS1.hdf5",
                            NG = 32,
                            )
pic3.two_stream_instability("data_analysis/TS2.hdf5",
                             NG = 64,
                             )
pic3.two_stream_instability("data_analysis/TS3.hdf5",
                            plasma_frequency=10,
                            N_electrons=int(1e5),
                            )
pic3.two_stream_instability("data_analysis/TS4.hdf5",
                            NT=9000,
                            )
pic3.two_stream_instability("data_analysis/TS5.hdf5",
                            NT=1000,
                            plasma_frequency=1.1*2**-0.5,
                            N_electrons=int(1e5),
                            )
pic3.two_stream_instability("data_analysis/TS6.hdf5",
                            NT=1000,
                            plasma_frequency=1.1*2**-0.5,
                            N_electrons=int(1e5),
                            NG=128,
                            )
pic3.two_stream_instability("data_analysis/TS7.hdf5",
                            NT=1000,
                            plasma_frequency=10,
                            N_electrons=int(1e5),
                            )
pic3.two_stream_instability("data_analysis/TS8.hdf5",
                            NT=1000,
                            plasma_frequency=10,
                            N_electrons=int(1e5),
                            NG=128,
                            )
pic3.two_stream_instability("data_analysis/TS9.hdf5",
                            NT=1000,
                            plasma_frequency=10,
                            N_electrons=int(1e5),
                            NG=256,
                            )
pic3.two_stream_instability("data_analysis/TS10.hdf5",
                            NT=1000,
                            plasma_frequency=10,
                            N_electrons=int(1e5),
                            NG=512,
                            )

show = False
plotting.plotting("data_analysis/TS1.hdf5", show=show)
plotting.plotting("data_analysis/TS2.hdf5", show=show)
plotting.plotting("data_analysis/TS3.hdf5", show=show)
plotting.plotting("data_analysis/TS4.hdf5", show=show)
plotting.plotting("data_analysis/TS5.hdf5", show=show)
plotting.plotting("data_analysis/TS6.hdf5", show=show)
plotting.plotting("data_analysis/TS7.hdf5", show=show)
plotting.plotting("data_analysis/TS8.hdf5", show=show)
plotting.plotting("data_analysis/TS9.hdf5", show=show)
plotting.plotting("data_analysis/TS10.hdf5", show=show)
