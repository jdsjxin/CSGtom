
CSGtom - A dynamic modelling platform for Chinese solar greenhouse with tomato crop
=========================================

A framework in Python for defining and running dynamic CSGtom models.

## Cite as
Bo Zhou, Daniel Reyes Lastiri, Nan Wang, Qichang Yang, Eldert J. van Henten. An opensource indoor climate and yield prediction model for Chinese solar greenhouses. Biosystems Engineering, 250(2025): 183-212.

## Current release
CSGtom v1.0.1 - Concise code without detailed explanations.

## Author and Maintainer
* Bo Zhou（周波）, `zhoubo02@caas.cn`, `zhb_nash@foxmail.com`

## In this repository

- `README.md`  - Readme file and supporting images
- `COPYING.txt` - The license for this repository (GNU GENERAL PUBLIC LICENSE, Version 3)
- *Code*
   - *classes* - Files for the ode module
      - `module.py` - Definition of the ode module 
   - *data* - Files for the data
      - `example_data.xls` - Input data file. This includes a two-day outdoor weather data example for model execution
      - `example_u.xls` - An example input file for control parameters
   - *functions* - Files for functions
      - `BVtomato_fun.py` - Tomato model function
      - `csg_fun.py` - Basic function for CSGtom
      - `csg_shape.py` - CSG shape function
      - `integration.py` - The Euler method
   - *models* - Files for main function
      - `csg_climate.py` - Basic example for CSGtom model 
   - *parameters* - Files for model parameters
      - `example.py` - An example of parameters

## License

This project is licensed under the GNU General Public License (GPL) Version 3, 29 June 2007 - see the [COPYING.txt](./COPYING.txt) file for details.

### Key points of the GPL v3:

- You can freely use, modify, and distribute the software, but you must keep the GPL v3 license and copyright notice intact.
- If you modify the code and distribute it (either in source or binary form), you must make the source code available to others.
- If you redistribute the software or its modified versions, you must do so under the same GPL v3 license.
- There is no warranty for the software. It is provided "as-is".

For more information, see the full text of the GPL v3 at [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html).
