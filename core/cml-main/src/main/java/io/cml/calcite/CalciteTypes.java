/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.cml.calcite;

import com.google.common.collect.ImmutableMap;
import io.cml.spi.CmlException;
import org.apache.calcite.sql.type.SqlTypeName;

import java.util.Map;
import java.util.Optional;

import static io.cml.spi.metadata.StandardErrorCode.NOT_SUPPORTED;
import static io.cml.spi.type.StandardTypes.ARRAY;
import static io.cml.spi.type.StandardTypes.BIGINT;
import static io.cml.spi.type.StandardTypes.BOOLEAN;
import static io.cml.spi.type.StandardTypes.BYTEA;
import static io.cml.spi.type.StandardTypes.CHAR;
import static io.cml.spi.type.StandardTypes.DATE;
import static io.cml.spi.type.StandardTypes.DECIMAL;
import static io.cml.spi.type.StandardTypes.DOUBLE;
import static io.cml.spi.type.StandardTypes.INTEGER;
import static io.cml.spi.type.StandardTypes.INTERVAL_DAY_TO_SECOND;
import static io.cml.spi.type.StandardTypes.INTERVAL_YEAR_TO_MONTH;
import static io.cml.spi.type.StandardTypes.REAL;
import static io.cml.spi.type.StandardTypes.SMALLINT;
import static io.cml.spi.type.StandardTypes.TIME;
import static io.cml.spi.type.StandardTypes.TIMESTAMP;
import static io.cml.spi.type.StandardTypes.TIMESTAMP_WITH_TIME_ZONE;
import static io.cml.spi.type.StandardTypes.TIME_WITH_TIME_ZONE;
import static io.cml.spi.type.StandardTypes.TINYINT;
import static io.cml.spi.type.StandardTypes.VARCHAR;
import static java.util.Locale.ROOT;

public final class CalciteTypes
{
    private CalciteTypes() {}

    private static final Map<String, SqlTypeName> standardTypeToCalciteTypeMap;

    static {
        standardTypeToCalciteTypeMap = ImmutableMap.<String, SqlTypeName>builder()
                .put(BIGINT, SqlTypeName.BIGINT)
                .put(INTEGER, SqlTypeName.INTEGER)
                .put(SMALLINT, SqlTypeName.SMALLINT)
                .put(TINYINT, SqlTypeName.TINYINT)
                .put(BOOLEAN, SqlTypeName.BOOLEAN)
                .put(DATE, SqlTypeName.DATE)
                .put(DECIMAL, SqlTypeName.DECIMAL)
                .put(REAL, SqlTypeName.REAL)
                .put(DOUBLE, SqlTypeName.DOUBLE)
                .put(INTERVAL_DAY_TO_SECOND, SqlTypeName.INTERVAL_DAY_SECOND)
                .put(INTERVAL_YEAR_TO_MONTH, SqlTypeName.INTERVAL_YEAR_MONTH)
                .put(TIMESTAMP, SqlTypeName.TIMESTAMP)
                // TODO: check timestamp with tz
                .put(TIMESTAMP_WITH_TIME_ZONE, SqlTypeName.TIME_WITH_LOCAL_TIME_ZONE)
                .put(TIME, SqlTypeName.TIME)
                // TODO: check time with tz
                .put(TIME_WITH_TIME_ZONE, SqlTypeName.TIME_WITH_LOCAL_TIME_ZONE)
                .put(BYTEA, SqlTypeName.VARBINARY)
                .put(VARCHAR, SqlTypeName.VARCHAR)
                .put(CHAR, SqlTypeName.CHAR)
                .put(ARRAY, SqlTypeName.ARRAY)
                .build();
    }

    public static SqlTypeName toCalciteType(String typeName)
    {
        return Optional.ofNullable(standardTypeToCalciteTypeMap.get(typeName.toLowerCase(ROOT)))
                .orElseThrow(() -> new CmlException(NOT_SUPPORTED, "Unsupported Type: " + typeName));
    }
}