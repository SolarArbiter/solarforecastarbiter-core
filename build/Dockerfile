FROM  docker.io/library/python:3.7-buster

WORKDIR /opt/app-root/
ENV PATH=/opt/app-root/bin:/usr/local/texlive/bin/x86_64-linux:$PATH \
    SUMMARY="All in one image for solarforecastarbiter-core including GPG, wgrib2, pdflatex" \
    LANG=en_US.UTF-8 \
    LC_ALL=C

LABEL summary="$SUMMARY" \
    maintainer="atlorenzo@email.arizona.edu"

USER root

COPY build/grib2 /tmp/grib2

RUN set -ex \
    && apt-get update \
    && apt-get upgrade -y \
    && buildDeps='gfortran libnetcdf-dev libhdf5-dev libgrib2c-dev' \
    && apt-get install -y $buildDeps python3-gpg libnetcdf13 sudo \
    && cd /root \
    # wgrib2 install, with help from fedora RPM build https://src.fedoraproject.org/rpms/wgrib2/tree/master
    && wget http://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2_nolib.tgz.v2.0.8 \
    && tar -xzf wgrib2_nolib.tgz.v2.0.8 \
    && cd grib2 \
    && /bin/bash -c "rm -r wgrib2/{fnlist,Gctpc,gctpc_ll2xy,new_grid_lambertc}.[ch]" \
    && patch -p1 -b < /tmp/grib2/wgrib2-jasper-2.patch \
    && patch -p1 -b < /tmp/grib2/wgrib2-nogctpc.patch \
    && cp /tmp/grib2/config.h /root/grib2/wgrib2/. \
    && cd wgrib2 \
    && CC=gcc FC=gfortran CFLAGES="-I.. -I/usr/include/netcdf" LDFLAGS="-lnetcdf -lpng -lz -lm" make fnlist.c fnlist.h\
    && CC=gcc FC=gfortran CFLAGES="-I.. -I/usr/include/netcdf" LDFLAGS="-lnetcdf -lpng -lz -lm" make \
    && mkdir -p /opt/app-root/bin \
    && cp wgrib2 /opt/app-root/bin \
    && cd /opt/app-root/ \
    && rm -rf /root/grib2 wgrib2.tgz /tmp/grib2 \
    && apt-get autoremove -y $buildDeps \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/local/bin/python -m venv /opt/app-root/ \
    && /opt/app-root/bin/pip install -U pip \
    && ln -s "/usr/lib/python3/dist-packages/gpg" "/opt/app-root/lib/python3.7/site-packages/" \
    && ln -s "/usr/lib/python3/dist-packages/gpg-*.egg-info" "/opt/app-root/lib/python3.7/site-packages/" \
    && /opt/app-root/bin/python -c "import gpg" \
    && groupadd wheel \
    && useradd -M -N -u 1001 -s /bin/bash -g 0 -G wheel user \
    && echo '%wheel   ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/wheel \
    && chmod 0440 /etc/sudoers.d/wheel \
    && chown -R 1001:0 /opt/app-root

COPY build/tex /tmp/tex

# install texlive basic
RUN set -ex \
    && cd /tmp/tex \
    && wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
    && mkdir install-tl \
    && tar -xzf install-tl-unx.tar.gz -C install-tl --strip=1 \
    && install-tl/install-tl -no-gui -profile /tmp/tex/texlive.profile \
    && mkdir -p /opt/app-root/texmf \
    && chown 1001:0 /opt/app-root/texmf \
    && su user -c "tlmgr init-usertree" \
    && rm -r /tmp/tex

USER 1001

COPY . /tmp/build
RUN set -ex \
    && tlmgr --usermode install $(cat /tmp/build/packages.tex) \
    && /opt/app-root/bin/pip install --no-cache-dir wheel \
    && /opt/app-root/bin/pip install --no-cache-dir -r /tmp/build/requirements.txt -r /tmp/build/requirements-test.txt \
    && /opt/app-root/bin/pip install --no-cache-dir /tmp/build/ \
    && rm -rf /tmp/build

CMD /opt/app-root/bin/solararbiter
