def update_observation_data(api, sites, observations, start, end):
    """Post new observation data to a list of Surfrad Observations
    from start to end.

    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    sites: list
        List of all reference sites as Objects
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    """
    sites = api.list_sites()
    surfrad_sites = filter(partial(common.check_network, 'NOAA SURFRAD'),
                           sites)
    for site in surfrad_sites:
        obs_df = fetch(api, site, start, end)
        site_observations = [obs for obs in observations if obs.site == site]
        for obs in site_observations:
            logger.info(
                f'Updating {obs.name} from '
                f'{obs_df.index[0].strftime("%Y%m%dT%H%MZ")} '
                f'to {obs_df.index[-1].strftime("%Y%m%dT%H%MZ")}.')
            var_df = obs_df[[obs.variable]]
            var_df = var_df.rename(columns={obs.variable: 'value'})
            var_df['quality_flag'] = 0 
            var_df = var_df.dropna()
            api.post_observation_values(obs.observation_id, var_df)
