OBJECTS_TO_ENV = {
    "button": [
        "button_press_topdown_v2",
        "button_press_topdown_wall_v2",
        "button_press_v2",
        "button_press_wall_v2",
    ],
    "basketball": ["basketball_v2"],
    "round_nut": ["assembly_v2", "disassemble_v2"],
    "block": [
        "bin_picking_v2",
        "hand_insert_v2",
        "pick_out_of_hole_v2",
        "pick_place_v2",
        "pick_place_wall_v2",
        "push_back_v2",
        "push_v2",
        "push_wall_v2",
        "shelf_place_v2",
        "sweep_into_v2",
        "sweep_v2",
    ],
    "top_link": ["box_close_v2"],
    "coffee_button_start": ["coffee_button_v2"],
    "coffee_mug": ["coffee_pull_v2", "coffee_push_v2"],
    "dial": ["dial_turn_v2"],
    "door": ["door_close_v2", "door_open_v2"],  # can we merge these together?
    "door_link": ["door_lock_v2", "door_unlock_v2", "door_v2"],
    "drawer_link": ["drawer_close_v2", "drawer_open_v2"],
    "faucet_handle": ["faucet_close_v2", "faucet_open_v2"],
    "hammer": ["hammer_v2"],
    "handle": [
        "handle_press_side_v2",
        "handle_press_v2",
        "handle_pull_side_v2",
        "handle_pull_v2",
        "lever_pull_v2",
    ],
    "peg": ["peg_insert_side_v2", "peg_unplug_side_v2"],
    "puck": [
        "plate_slide_back_side_v2",
        "plate_slide_back_v2",
        "plate_slide_side_v2",
        "plate_slide_v2",
    ],
    "no_obj": ["reach_v2", "reach_wall_v2"],
    "shelf": ["shelf_place_v2"],
    "soccer_ball": ["soccer_v2"],
    "stick": ["stick_push_v2", "stick_pull_v2"],
    "window_handle": ["window_close_v2", "window_open_v2"],
}

OBJECTS = list(OBJECTS_TO_ENV.keys())
OBJECTS = ["no_obj"] + OBJECTS

ENV_TO_OBJECTS = {}
for obj, envs in OBJECTS_TO_ENV.items():
    for env in envs:
        if env not in ENV_TO_OBJECTS:
            ENV_TO_OBJECTS[env] = [obj]
        else:
            ENV_TO_OBJECTS[env].append(obj)
