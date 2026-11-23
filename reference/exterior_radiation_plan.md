# Exterior radiation implementation plan

1. Add optional exterior radiation to `FloorPlanBasedBuilding`. Reuse its
    exterior-wall optical properties and accept glazing optical properties.
    Derive exposed vertical face areas from the existing CV geometry and
    resolve the four compass orientations independently.
2. Use `WeatherController` and `ReplayWeatherController` to supply one weather
    sample per timestep. Reuse `SolarRadiation` for sky temperature and
    plane-of-array irradiance. Use outdoor air temperature for ground
    temperature when a separate measurement is unavailable.
3. Assemble exterior longwave, absorbed solar, and transmitted solar powers in
    watts. Deposit the first two at the exposed CV. Distribute transmitted
    power once across the connected indoor air region, routing shares to the
    interior mass where present. Use the exposed CV temperature as the discrete
    surface temperature in the longwave equation.
4. Add those powers to the scalar, TensorFlow, and JAX balances. At each Picard
    sweep, factor the fourth-power difference into a conductance and
    temperature difference. Put the conductance on the solver diagonal and
    evaluate it on-device for resident solvers. Keep HVAC input separate.
5. Run test-first checks for the weather inputs, cooling and heating signs,
    corner face orientation and area, transmitted power, and solver parity. Run
    the existing simulator tests after the new tests pass.

References: `buildsys2026_tensor.pdf`, sections 3.1–3.4, and
`sbsim_equation.pdf`, sections 5–7.

## Configuration

Set `include_exterior_radiative_heat_transfer=True` on `FloorPlanBasedBuilding`.
Set `floor_plan_orientation` to the clockwise bearing of the grid's up
direction, and use value 4 for glazing in the floor plan. Connect transmitting
glazing to indoor air through its window cells.
`building_exterior_radiative_properties` and `fenestration_radiative_properties`
control the optical coefficients. Interior longwave exchange and interior mass
remain independently configurable.

Pass a `SolarRadiation` instance configured with the building's location to
`WeatherController` or `ReplayWeatherController` as `solar_radiation_model`. Use
the building's `floor_plan_orientation` for facade rotation and the weather
model's `BuildingInfo` for location and timezone. The replay controller uses
measured `DewPointC` for sky temperature and `SkyCoverage` for modeled
irradiance when those readings are available. A custom weather controller can
supply measured irradiance, sky temperature, or ground temperature through
`get_exterior_radiation`. Weather is sampled at the start of
`finite_differences_timestep` and its calling step methods.
