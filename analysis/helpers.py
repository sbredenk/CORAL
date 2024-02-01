import datetime as dt

target_capacity = {
    '25 GW': 25,
    '35 GW': 35,
    '55 GW': 55
}

allocations = {
    # '25 GW - Low': {
        # "ahts_vessel": ('example_ahts_vessel', 1), # =3 AHTS (1 projects at a time, 3 groups per project, 1 AHTS per group)
        # "towing_vessel": ('example_towing_vessel', 1), # =6 tugs (1 projects at a time, 3 groups per project, 2 tugs per group)
        # # "mooring_install_vessel": ('example_support_vessel', 1), # =1 AHTS (1 projects at a time, 1 per project)
        # # "array_cable_install_vessel": ('example_array_cable_lay_vessel', 1), # =1 CLVs (1 projects at a time, 1 CLV per project)
        # # "export_cable_install_vessel": ("example_export_cable_lay_vessel", 1), # =1 CLVs (1 projects at a time, 1 CLV per project)
        # "port": [('Humboldt', 1)]
            # },
    ## Saturate # of vessels to iterate on port constraints and wait for data on actual numbers
    '25 GW (SC)': {
        "ahts_vessel": ('example_ahts_vessel', 3), # =9 AHTS
        "towing_vessel": ('example_towing_vessel', 3), # =18 tugs
        # "mooring_install_vessel": ('example_support_vessel', 3), # =3 AHTS
        # "array_cable_install_vessel": ('example_array_cable_lay_vessel', 3), # =3 CLV (array)
        # "export_cable_install_vessel": ("example_export_cable_lay_vessel", 3), # =3 CLV (export)
        "port": [('Long Beach', 0), ('Humboldt', 1)]
            },
    '25 GW (CC)': {
        "ahts_vessel": ('example_ahts_vessel', 2), # =6 AHTS
        "towing_vessel": ('example_towing_vessel', 2), # =12 tugs
        # "mooring_install_vessel": ('example_support_vessel', 2), # =2 AHTS
        # "array_cable_install_vessel": ('example_array_cable_lay_vessel', 2), # =2 CLV (array)
        # "export_cable_install_vessel": ("example_export_cable_lay_vessel",2), # =2 CLV (export)
        "port": [('Port of San Luis', 0), ('Humboldt', 1)]
            },
    # 'Moderate-Low': {
        # "ahts_vessel": ('example_ahts_vessel', 3), # =9 AHTS
        # "towing_vessel": ('example_towing_vessel', 3), # =18 tugs
        # # "mooring_install_vessel": ('example_support_vessel', 3), # =3 AHTS
        # # "array_cable_install_vessel": ('example_array_cable_lay_vessel', 3), # =3 CLV (array)
        # # "export_cable_install_vessel": ("example_export_cable_lay_vessel",3), # =3 CLV (export)
        # "port": [('Humboldt', 1), ('Coos Bay', 0)]
            # },
    '35 GW': {
        "ahts_vessel": ('example_ahts_vessel', 4), # =12 AHTS
        "towing_vessel": ('example_towing_vessel', 4), # =24 tugs
        # "mooring_install_vessel": ('example_support_vessel', 4), # =4 AHTS
        # "array_cable_install_vessel": ('example_array_cable_lay_vessel', 4), # =4 CLV (array)
        # "export_cable_install_vessel": ("example_export_cable_lay_vessel",4), # =4 CLV (export)
        "port": [('Long Beach', 0), ('Humboldt', 1), ('Coos Bay', 0)]
            },
    '55 GW': {
        "ahts_vessel": ('example_ahts_vessel', 8), # =24 AHTS
        "towing_vessel": ('example_towing_vessel', 8), # =48 tugs
        # "mooring_install_vessel": ('example_support_vessel', 8), # =8 AHTS
        # "array_cable_install_vessel": ('example_array_cable_lay_vessel', 8), # =8 CLV (array)
        # "export_cable_install_vessel": ("example_export_cable_lay_vessel", 8), # =8 CLV (export)
        "port": [('Long Beach', 0), ('Humboldt', 1), ('Coos Bay', 0), ('Grays Harbor', 0), ('Port of San Luis', 0)]
            }
}

future_allocations = {
    # '25 GW - Low': [
        # ["port", "Humboldt", [dt.datetime(2030, 1, 1)]]
    # ],
    '25 GW (SC)': [
        ["port", "Long Beach", [dt.datetime(2031, 1, 1)]],
        ["port", "Humboldt", [dt.datetime(2030, 1, 1)]],
        ["port", "Long Beach", [dt.datetime(2031, 1, 1)]]
    ],
    '25 GW (CC)': [
        ["port", "Port of San Luis", [dt.datetime(2037, 1, 1)]],
        ["port", "Humboldt", [dt.datetime(2030, 1, 1)]]
    ],
    # 'Moderate-Low':[
        # ["port", "Coos Bay", [dt.datetime(2031, 1, 1)]],
        # ["port", "Humboldt", [dt.datetime(2030, 1, 1)]],
        # ["port", "Coos Bay", [dt.datetime(2038, 1, 1)]]
    # ],
    '35 GW':[
        ["port", "Long Beach", [dt.datetime(2031, 1, 1)]],
        ["port", "Long Beach", [dt.datetime(2031, 1, 1)]],
        ["port", "Coos Bay", [dt.datetime(2031, 1, 1)]],
        ["port", "Humboldt", [dt.datetime(2030, 1, 1)]],
        ["port", "Coos Bay", [dt.datetime(2038, 1, 1)]]
    ],
    '55 GW':[
        ["port", "Humboldt", [dt.datetime(2030, 1, 1)]],
        ["port", "Long Beach", [dt.datetime(2031, 1, 1)]],
        ["port", "Long Beach", [dt.datetime(2031, 1, 1)]], # Add 2nd line
        # ["port", "Long Beach", [dt.datetime(2035, 1, 1)]], # Add 3rd line
        ["port", "Port of San Luis", [dt.datetime(2037, 1, 1)]],
        ["port", "Coos Bay", [dt.datetime(2031, 1, 1)]],
        ["port", "Coos Bay", [dt.datetime(2035, 1, 1)]], # Accelerate from 2038
        ["port", "Grays Harbor", [dt.datetime(2035, 1, 1)]], # Accelerate from 2039
        ["port", "Grays Harbor", [dt.datetime(2035, 1, 1)]], # Add 2nd line
        #["port", "Grays Harbor", [dt.datetime(2035, 1, 1)]], # Add 3rd line/2nd port
    ]
}
