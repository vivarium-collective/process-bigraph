
# Documentation: https://devenv.sh/

{ pkgs, lib, config, inputs, ... }:

let
  buildInputs = with pkgs; [ stdenv.cc.cc libuv zlib ];
in

{
  env = {
    LD_LIBRARY_PATH = "${with pkgs; lib.makeLibraryPath buildInputs}";
  };

  packages = with pkgs; [
    git
    (python312.withPackages (py: with py;
      [ ipython jupyter ]
    ))
  ];

  languages.python = {
    enable = true;
    version = "3.12.9";
    uv = {
      enable = true;
      sync.enable = true;
    };
  };
  languages.julia.enable = true;

  enterShell = ''
    . .devenv/state/venv/bin/activate
  '';
}
