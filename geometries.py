import numpy as np
import random
from itertools import product


from fdm_solver import (
    get_w_strain,
    draw_circle,
    draw_rect,
    draw_ellipse,
    draw_parabola_region,
    apply_swg_mask,
)
Ny, Nx = 256, 256
cx, cy = Nx // 2, Ny // 2
SCENARIOS = {

                                                                               
    'single_circle_center': lambda m: draw_circle(
        m, cx, cy, r=random.randint(20, 60)),

    'single_circle_edge': lambda m: draw_circle(
        m, random.choice([25, Nx - 25]), cy, r=random.randint(15, 35)),

    'single_rect_center': lambda m: draw_rect(
        m, cx, cy,
        w=random.randint(30, 100), h=random.randint(30, 100),
        angle_deg=random.uniform(0, 90)),

    'single_ellipse': lambda m: draw_ellipse(
        m, cx, cy,
        rx=random.randint(20, 70), ry=random.randint(10, 40),
        angle_deg=random.uniform(0, 180)),

                                                                               
    'two_circles_side': lambda m: [
        draw_circle(m, cx - random.randint(40, 70), cy, random.randint(15, 35)),
        draw_circle(m, cx + random.randint(40, 70), cy, random.randint(15, 35))],

    'two_circles_close': lambda m: [
        draw_circle(m, cx - random.randint(15, 30), cy, random.randint(15, 25)),
        draw_circle(m, cx + random.randint(15, 30), cy, random.randint(15, 25))],

    'circle_and_rect': lambda m: [
        draw_circle(m, cx - 40, cy, random.randint(15, 30)),
        draw_rect(m, cx + 40, cy,
                  random.randint(20, 50), random.randint(20, 50))],

    'grid_circles': lambda m: [
        draw_circle(m, x, y, random.randint(10, 20))
        for x, y in product(
            range(40, Nx - 40, random.randint(40, 70)),
            range(40, Ny - 40, random.randint(40, 70)))],

    'row_circles': lambda m: [
        draw_circle(m, x, cy, random.randint(10, 25))
        for x in range(40, Nx - 40, random.randint(35, 60))],

    'random_mix': lambda m: [
        (lambda shape, cx_, cy_:
            draw_circle(m, cx_, cy_, random.randint(10, 40))
            if shape == 'circle' else
            draw_ellipse(m, cx_, cy_, random.randint(10, 40),
                         random.randint(10, 30), random.uniform(0, 180))
         )(random.choice(['circle', 'ellipse']),
           random.randint(30, Nx - 30),
           random.randint(30, Ny - 30))
        for _ in range(random.randint(2, 6))],

                                                                               

                                          
    'parabola_concave_up': lambda m: draw_parabola_region(
        m, cx, cy,
        amplitude=random.uniform(0.2, 0.7),
        rotation_deg=0,
        fill='below'),

                                         
    'parabola_convex_up': lambda m: draw_parabola_region(
        m, cx, cy,
        amplitude=random.uniform(0.2, 0.7),
        rotation_deg=0,
        fill='above'),

                                 
    'parabola_concave_rotated': lambda m: draw_parabola_region(
        m, cx, cy,
        amplitude=random.uniform(0.2, 0.7),
        rotation_deg=random.uniform(0, 360),
        fill='below'),

                                
    'parabola_convex_rotated': lambda m: draw_parabola_region(
        m, cx, cy,
        amplitude=random.uniform(0.2, 0.7),
        rotation_deg=random.uniform(0, 360),
        fill='above'),

                               
    'parabola_and_circle': lambda m: [
        draw_parabola_region(
            m, cx, cy,
            amplitude=random.uniform(0.2, 0.7),
            rotation_deg=random.uniform(0, 360),
            fill=random.choice(['below', 'above'])),
        draw_circle(m,
                    random.randint(30, Nx - 30),
                    random.randint(30, Ny - 30),
                    random.randint(10, 25))],

                            
    'parabola_and_2circles': lambda m: [
        draw_parabola_region(
            m, cx, cy,
            amplitude=random.uniform(0.2, 0.7),
            rotation_deg=random.uniform(0, 360),
            fill=random.choice(['below', 'above'])),
        draw_circle(m, cx - random.randint(20, 60), cy, random.randint(10, 20)),
        draw_circle(m, cx + random.randint(20, 60), cy, random.randint(10, 20))],

                        
    'parabola_and_ellipse': lambda m: [
        draw_parabola_region(
            m, cx, cy,
            amplitude=random.uniform(0.2, 0.7),
            rotation_deg=random.uniform(0, 360),
            fill=random.choice(['below', 'above'])),
        draw_ellipse(m,
                     random.randint(30, Nx - 30),
                     random.randint(30, Ny - 30),
                     rx=random.randint(10, 40),
                     ry=random.randint(10, 30),
                     angle_deg=random.uniform(0, 180))],

                                                                               
     'swg': lambda m: apply_swg_mask(m),
                         
     'swg_and_parabola': lambda m: [
        apply_swg_mask(m),                                                                        
        draw_parabola_region(                                                      
            m, cx, cy,
            amplitude=random.uniform(0.2, 0.7),
            rotation_deg=random.uniform(0, 360),
            fill=random.choice(['below', 'above']))],

                       
     'swg_and_circle': lambda m: [
        apply_swg_mask(m),
        draw_circle(m,
                    random.randint(30, Nx - 30),
                    random.randint(30, Ny - 30),
                    random.randint(15, 50))],

                        
    'swg_and_ellipse': lambda m: [
        apply_swg_mask(m),
        draw_ellipse(m,
                     random.randint(30, Nx - 30),
                     random.randint(30, Ny - 30),
                     rx=random.randint(15, 55),
                     ry=random.randint(10, 35),
                     angle_deg=random.uniform(0, 180))],
}