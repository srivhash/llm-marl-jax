def featurize_state(self, overcooked_state, mlam, num_pots=2, **kwargs):
        """
        Encode state with some manually designed features. Works for arbitrary number of players

        Arguments:
            overcooked_state (OvercookedState): state we wish to featurize
            mlam (MediumLevelActionManager): to be used for distance computations necessary for our higher-level feature encodings
            num_pots (int): Encode the state (ingredients, whether cooking or not, etc) of the 'num_pots' closest pots to each player.
                If i < num_pots pots are reachable by player i, then pots [i+1, num_pots] are encoded as all zeros. Changing this
                impacts the shape of the feature encoding

        Returns:
            ordered_features (list[np.Array]): The ith element contains a player-centric featurized view for the ith player

            The encoding for player i is as follows:

                [player_i_features, other_player_features player_i_dist_to_other_players, player_i_position]

                player_{i}_features (length num_pots*10 + 28):
                    pi_orientation: length 4 one-hot-encoding of direction currently facing
                    pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
                    pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
                    pi_cloest_soup_n_{onions|tomatoes}: int value for number of this ingredient in closest soup
                    pi_closest_pot_{j}_exists: {0, 1} depending on whether jth closest pot found. If 0, then all other pot features are 0. Note: can
                        be 0 even if there are more than j pots on layout, if the pot is not reachable by player i
                    pi_closest_pot_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot
                    pi_closest_pot_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest pot
                    pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
                    pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location
                    pi_wall: length 4 boolean value of whether player i has wall in each direction

                other_player_features (length (num_players - 1)*(num_pots*10 + 24)):
                    ordered concatenation of player_{j}_features for j != i

                player_i_dist_to_other_players (length (num_players - 1)*2):
                    [player_j.pos - player_i.pos for j != i]

                player_i_position (length 2)
        """

        all_features = {}

        def concat_dicts(a, b):
            return {**a, **b}

        def make_closest_feature(idx, player, name, locations):
            """
            Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict
            """
            feat_dict = {}
            obj = None
            held_obj = player.held_object
            held_obj_name = held_obj.name if held_obj else "none"
            if held_obj_name == name:
                obj = held_obj
                feat_dict["p{}_closest_{}".format(i, name)] = (0, 0)
            else:
                loc, deltas = self.get_deltas_to_closest_location(
                    player, locations, mlam
                )
                if loc and overcooked_state.has_object(loc):
                    obj = overcooked_state.get_object(loc)
                feat_dict["p{}_closest_{}".format(idx, name)] = deltas

            if name == "soup":
                num_onions = num_tomatoes = 0
                if obj:
                    ingredients_cnt = Counter(obj.ingredients)
                    num_onions, num_tomatoes = (
                        ingredients_cnt["onion"],
                        ingredients_cnt["tomato"],
                    )
                feat_dict["p{}_closest_soup_n_onions".format(i)] = [num_onions]
                feat_dict["p{}_closest_soup_n_tomatoes".format(i)] = [
                    num_tomatoes
                ]

            return feat_dict

        def make_pot_feature(idx, player, pot_idx, pot_loc, pot_states):
            """
            Encode pot at pot_loc relative to 'player'
            """
            # Pot doesn't exist
            feat_dict = {}
            if not pot_loc:
                feat_dict["p{}_closest_pot_{}_exists".format(idx, pot_idx)] = [
                    0
                ]
                feat_dict[
                    "p{}_closest_pot_{}_is_empty".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_full".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_cooking".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_ready".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_num_onions".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_num_tomatoes".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_cook_time".format(idx, pot_idx)
                ] = [0]
                feat_dict["p{}_closest_pot_{}".format(idx, pot_idx)] = (0, 0)
                return feat_dict

            # Get position information
            deltas = self.get_deltas_to_location(player, pot_loc)

            # Get pot state info
            is_empty = int(pot_loc in self.get_empty_pots(pot_states))
            is_full = int(pot_loc in self.get_full_pots(pot_states))
            is_cooking = int(pot_loc in self.get_cooking_pots(pot_states))
            is_ready = int(pot_loc in self.get_ready_pots(pot_states))

            # Get soup state info
            num_onions = num_tomatoes = 0
            cook_time_remaining = 0
            if not is_empty:
                soup = overcooked_state.get_object(pot_loc)
                ingredients_cnt = Counter(soup.ingredients)
                num_onions, num_tomatoes = (
                    ingredients_cnt["onion"],
                    ingredients_cnt["tomato"],
                )
                cook_time_remaining = (
                    0 if soup.is_idle else soup.cook_time_remaining
                )

            # Encode pot and soup info
            feat_dict["p{}_closest_pot_{}_exists".format(idx, pot_idx)] = [1]
            feat_dict["p{}_closest_pot_{}_is_empty".format(idx, pot_idx)] = [
                is_empty
            ]
            feat_dict["p{}_closest_pot_{}_is_full".format(idx, pot_idx)] = [
                is_full
            ]
            feat_dict["p{}_closest_pot_{}_is_cooking".format(idx, pot_idx)] = [
                is_cooking
            ]
            feat_dict["p{}_closest_pot_{}_is_ready".format(idx, pot_idx)] = [
                is_ready
            ]
            feat_dict["p{}_closest_pot_{}_num_onions".format(idx, pot_idx)] = [
                num_onions
            ]
            feat_dict[
                "p{}_closest_pot_{}_num_tomatoes".format(idx, pot_idx)
            ] = [num_tomatoes]
            feat_dict["p{}_closest_pot_{}_cook_time".format(idx, pot_idx)] = [
                cook_time_remaining
            ]
            feat_dict["p{}_closest_pot_{}".format(idx, pot_idx)] = deltas

            return feat_dict

        IDX_TO_OBJ = ["onion", "soup", "dish", "tomato"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_states = self.get_pot_states(overcooked_state)

        for i, player in enumerate(overcooked_state.players):
            # Player info
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[
                orientation_idx
            ]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[
                    obj_idx
                ]

            # Closest feature for each object type
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "onion",
                    self.get_onion_dispenser_locations()
                    + counter_objects["onion"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "tomato",
                    self.get_tomato_dispenser_locations()
                    + counter_objects["tomato"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "dish",
                    self.get_dish_dispenser_locations()
                    + counter_objects["dish"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i, player, "soup", counter_objects["soup"]
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i, player, "serving", self.get_serving_locations()
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "empty_counter",
                    self.get_empty_counter_locations(overcooked_state),
                ),
            )

            # Closest pots info
            pot_locations = self.get_pot_locations().copy()
            for pot_idx in range(num_pots):
                _, closest_pot_loc = mlam.motion_planner.min_cost_to_feature(
                    player.pos_and_or, pot_locations, with_argmin=True
                )
                pot_features = make_pot_feature(
                    i, player, pot_idx, closest_pot_loc, pot_states
                )
                all_features = concat_dicts(all_features, pot_features)

                if closest_pot_loc:
                    pot_locations.remove(closest_pot_loc)

            # Adjacent features info
            for direction, pos_and_feat in enumerate(
                self.get_adjacent_features(player)
            ):
                _, feat = pos_and_feat
                all_features["p{}_wall_{}".format(i, direction)] = (
                    [0] if feat == " " else [1]
                )

        # Convert all list and tuple values to np.arrays
        features_np = {k: np.array(v) for k, v in all_features.items()}

        player_features = []  # Non-position player-specific features
        player_absolute_positions = []  # Position player-specific features
        player_relative_positions = (
            []
        )  # Relative position player-specific features

        # Compute all player-centric features for each player
        for i, player_i in enumerate(overcooked_state.players):
            # All absolute player-centric features
            player_i_dict = {
                k: v
                for k, v in features_np.items()
                if k[:2] == "p{}".format(i)
            }
            features = np.concatenate(list(player_i_dict.values()))
            abs_pos = np.array(player_i.position)

            # Calculate position relative to all other players
            rel_pos = []
            for player_j in overcooked_state.players:
                if player_i == player_j:
                    continue
                pj_rel_to_pi = np.array(
                    pos_distance(player_j.position, player_i.position)
                )
                rel_pos.append(pj_rel_to_pi)
            rel_pos = np.concatenate(rel_pos)

            player_features.append(features)
            player_absolute_positions.append(abs_pos)
            player_relative_positions.append(rel_pos)

        # Compute a symmetric, player-centric encoding of features for each player
        ordered_features = []
        for i, player_i in enumerate(overcooked_state.players):
            player_i_features = player_features[i]
            player_i_abs_pos = player_absolute_positions[i]
            player_i_rel_pos = player_relative_positions[i]
            other_player_features = np.concatenate(
                [feats for j, feats in enumerate(player_features) if j != i]
            )
            player_i_ordered_features = np.squeeze(
                np.concatenate(
                    [
                        player_i_features,
                        other_player_features,
                        player_i_rel_pos,
                        player_i_abs_pos,
                    ]
                )
            )
            ordered_features.append(player_i_ordered_features)

        return ordered_features