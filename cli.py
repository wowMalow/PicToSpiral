import click
from engine import transform_image


@click.command()
@click.option('-m', '--mode', default='PNG', help='PNG or SVG. Export result in chosen format')
@click.option('-a', '--alpha', default=3.0, help='From 0.5 to 20.0. Alpha increases spiral step')
@click.option('-bg1', '--background-color1', default='#640E05', help='HEX color. Top-left background color')
@click.option('-bg2', '--background-color2', default='#051646', help='HEX color. Bottom-right background color')
@click.option('-s', '--spiral-color', default='#65BFFC', help='HEX color of spiral')
@click.option('-i', '--inverse_flag', is_flag=True, help='Inverse image colors')
@click.option('-c', '--contrast_flag', is_flag=True, help='Add contrast')
@click.argument('file')
def main(**args):
    """
    This script transforms image into spiral vector image in PNG os SVG (by your choice).
    You can adjust spiral size and color, background linear gradient. Also you can
    add contrast to image or inverse colors.

    For work write path to file, or put it in program directory.
    """
    transform_image(**args)


if __name__ == '__main__':
    main()
