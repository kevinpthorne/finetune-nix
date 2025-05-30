{
  description = "Dev shell for fine-tuning DeepSeek on Nix documentation (Python 3.12)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };
  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;  # forces local compilation
        };

        python-with-packages = pkgs.python312.withPackages (ps: with ps; [
          pip
          setuptools
          wheel
          torch
          torchvision
          torchaudio
          transformers
          datasets
          accelerate
          beautifulsoup4
          requests
          GitPython
          peft
          bitsandbytes
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "deepseek-nix-finetuning";

          buildInputs = [
            python-with-packages
            pkgs.git
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cuda_nvcc
            pkgs.ncurses
            pkgs.which
            pkgs.stdenv.cc.cc.lib
          ];

	  shellHook = ''
	    echo "🐍 Setting up Python 3.12 environment for DeepSeek fine-tuning..."
	    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	    export TRANSFORMERS_CACHE="$PWD/.cache/huggingface"
	    export HF_HOME="$PWD/.cache/huggingface"
	    export CUDA_VISIBLE_DEVICES=0
            export CUDA_PATH=${pkgs.cudatoolkit}
	  '';

        };
      });
}

