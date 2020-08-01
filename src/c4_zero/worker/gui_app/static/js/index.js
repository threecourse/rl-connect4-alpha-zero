Vue.config.debug = true;
var app = new Vue({
    el: "#app",
    delimiters: ["[[", "]]"],
    data: {
        H: 6,
        W: 7,
        board:
            ".................................................................................",
        legal_moves:
            ".................................................................................",
        message: "",

        evaluated: 0,
        game_is_ready: 1,
        message2: "",
        q_values: null,
        n_values: null,
        v_values: "",
        p_values: "",
        current_value: "",

        board_state_string:
            "#W60#####\n" +
            "#.......#\n" +
            "#.......#\n" +
            "#.......#\n" +
            "#.......#\n" +
            "#.......#\n" +
            "#oxoxoxo#\n" +
            "#########",
    },
    created: function () {

    },
    methods: {
        // get attributes
        getCellImage: function (ri, ci) {
            var i = ri * this.W + ci;
            var cell = this.board[i];
            switch (cell) {
                case ".":
                    return "none";
                case "o":
                    return "maru";
                case "x":
                    return "batsu";
            }
        },
        getCellAttribute: function (ri, ci) {
            var i = ri * this.W + ci;
            var cell = this.legal_moves[i];
            switch (cell) {
                case ".":
                    return "legal-no";
                case "o": {
                    if (this.game_is_ready)
                        return "legal-yes-human";
                    else
                        return "legal-yes-cpu";
                }
            }
        },
        get_n_value: function (ri, ci) {
            if (this.evaluated === 0 || this.n_values === "")
                return "";
            else if (this.n_values[ri][ci] === 0)
                return "";
            else {
                return this.n_values[ri][ci];
            }
        },
        get_q_value: function (ri, ci) {
            if (this.evaluated === 0 || this.q_values === "")
                return "";
            else if (this.n_values[ri][ci] === 0)
                return "";
            else {
                return Math.round(this.q_values[ri][ci] * 100);
            }
        },
        get_v_value: function (ri, ci) {
            if (this.evaluated === 0 || this.v_values === "")
                return "";
            else if (this.n_values[ri][ci] === 0)
                return "";
            else {
                return Math.round(this.v_values[ri][ci] * 100);
            }
        },
        get_p_value: function (ri, ci) {
            if (this.evaluated === 0 || this.v_values === "")
                return "";
            else if (this.n_values[ri][ci] === 0)
                return "";
            else {
                return Math.round(this.p_values[ri][ci] * 100);
            }
        },
        // query data
        new_game: function () {
            var self = this;
            var data = {
                "cmd_type": "new_game",
                "cmd_data": "",
                "cmd_data2": "",
            };
            self.clear_message();
            self.game_is_ready = 0;
            self.post(data);
        },
        load_game: function () {
            var self = this;
            var data = {
                "cmd_type": "load_game",
                "cmd_data": "",
                "cmd_data2": this.board_state_string,
            };
            self.clear_message();
            self.game_is_ready = 0;
            self.post(data);
        },
        try_move: function (ri, ci) {
            var self = this;

            var data = {
                "cmd_type": "try_move",
                "cmd_data": ri + "," + ci,
                "cmd_data2": "",
            };
            self.game_is_ready = 0;
            self.clear_message();
            self.post(data);
        },
        think_ai: function () {
            var self = this;

            var data = {
                "cmd_type": "think_ai",
                "cmd_data": "",
                "cmd_data2": "",
            };
            this.game_is_ready = 0;
            self.post(data);
            self.wait_until_ready(data);
        },
        move_ai: function () {
            var self = this;

            var data = {
                "cmd_type": "move_ai",
                "cmd_data": "",
                "cmd_data2": "",
            };
            this.game_is_ready = 0;
            self.post(data);
            self.wait_until_ready(data);
        },
        clear_message: function () {
            self.message2 = "";
        },
        wait_until_ready: function () {
            var self = this;

            // wait until getting ready
            self.message2 = "thinking";
            var timer = setInterval(
                () => {
                    if (self.game_is_ready === 1) {
                        clearInterval(timer);
                        self.message2 = "V = " + self.v_current;
                    } else {
                        self.message2 += ".";
                    }
                }
                , 200
            );
        },
        post: function (data) {
            var self = this;
            fetch("/game", {
                method: "POST",
                credentials: "same-origin",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            }).then(function (resp) {
                var data = resp.json();
                return data;
            }).then(function (data) {
                self.game_is_ready = 1;
                self.evaluated = parseInt(data["evaluated"]);
                self.board = data["board"];
                self.message = data["message"];
                self.legal_moves = data["legal_moves"];
                self.q_values = data["q_values"];
                self.n_values = data["n_values"];
                self.p_values = data["p_values"];
                self.v_values = data["v_values"];
                self.v_current = data["v_current"];

                if (self.evaluated === 0) {
                    self.message2 = "";
                }
            });
        }
    }
});
