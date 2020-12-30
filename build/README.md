# Docker image

Docker images are available at [quay.io](https://quay.io/repository/solararbiter/solarforecastarbiter-core)

```
> docker pull quay.io/solararbiter/solarforecastarbiter-core
```

Run the image

```
> docker run --rm quay.io/solararbiter/solarforecastarbiter-core:latest                                                                                        127 ↵
Usage: solararbiter [OPTIONS] COMMAND [ARGS]...

  The SolarForecastArbiter core command line tool

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  fetchnwp             Retrieve weather forecasts with variables relevant
                       to...
  referenceaggregates  Updates reference data for the given period.
  referencedata        Tool for initializing and updating...
  referencefx          Make reference forecasts
  referencenwp         Make the reference NWP forecasts that should be...
  report               Make a report.
  test                 Test this installation of solarforecastarbiter
  validate             Run the validation tasks for a given set of...
```

Build the image from the root directory of the `solarforecastarbiter-core` repository

```
> docker build -f build/Dockerfile .
```

See the [Docker documentation](https://docs.docker.com/).
