# coding=utf-8


# @pytest.mark.parametrize(["N_beam", "should_it_thermalize"], [
#     (128, False),
#     (512, False),
#     (2048, False),
#     ])
# def test_twostream_likeness(N_beam, should_it_thermalize):
#     run_name = f"BP_TWOSTREAM_{N_beam}"
#     S = weakbeam_instability(run_name, N_beam=N_beam,
#                              save_data=False)
#     assert (did_it_thermalize(S)[:2] == should_it_thermalize).all(), ("Incorrect thermalization",
#                                                                   plotting(S, show=show_on_fail, save=False,
#                                                                            animate=True))
