# chessreader: a chessboard live analyzer

_chessreader_ uses the images captured from a webcam placed above a chessboard
to follow a live game of chess, played between two humans. The game can be
monitored _(check for illegal moves, checkmate, etc.)_ and more importantly
recorded.

_chessreader_ should also be able to connect to a chess engine and allow a human
to play against the engine on a real chessboard _(assuming the human moves the
pieces for the engine - can't do everything for you now, can we?)_

Using image recognition techniques, our goal is to be able to run on virtually
_any_ chessboard, with nothing more than a well-placed webcam, right above the
board _(e.g. hanging from the head of a desk lamp)_. Unlike other solutions like
[DGT e-boards](http://www.digitalgametechnology.com/), or
[chessboarduino](http://chessboarduino.blogspot.fr/), we don't want to place
RFIDs or any kind of sensors on your pieces or your chessboard.

## Current Development Status

_chessreader_ is still in its early development phase - any help will be greatly
appreciated! :)

Please take a look at the wiki for an insight on how _chessreader_ works under
the hood, and don't hesitate to contact us if you want to.

### Dependencies - Installation

Chessreader requires Python >= 3.4. See `requirements.txt` for the list of
python libraries required - you can install them in a
[virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) if you
feel like it.

## Usage

Well at the moment there's not much to do with it... Be patient! ;)

## Testing

_chessreader_ uses the [nosetests](https://nose.readthedocs.org/en/latest/)
framework: simply run `nosetests` from the base directory.

You can also run `python chessreader.py --test`

<!-- vim: set ft=markdown tw=80: -->
